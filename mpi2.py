"""
MPI version 2 - Parallel word counting with SIZE-AWARE (LPT greedy) distribution.

Why: in version 1 every process gets the same NUMBER of files, but file sizes
are very heterogeneous (60% small 2k-5k words, 30% medium 10k-20k, 10% large
50k-100k). The work is proportional to the number of tokens (~bytes), not to
the number of files, so some processes get much more work than others.

Fix: the file sizes are read with os.path.getsize (metadata only, the file
content is not read); this is split among all processes and gathered in rank 0.
Rank 0 sorts the files from largest to smallest and assigns each one to the
process with the fewest bytes assigned so far (Longest-Processing-Time-first
greedy). The counting itself is identical to v1.

Steps (Part 1 of the lab):
  1. rank 0 reads dataset/consulta.txt
  2. rank 0 broadcasts the query words to all processes (comm.bcast)
  3. rank 0 obtains the list of file_*.txt files (sorted by name)
  4. the files are distributed by SIZE with LPT greedy: every process gets
     about the same number of BYTES (comm.scatter)
  5. each process counts the query words in its files (same logic and names
     as baseline_secuencial.py)
  6. partial results are gathered in rank 0 (comm.gather)
  7. rank 0 builds the global result, prints the top 10, compares it with
     dataset/baseline_results.csv and prints EXECUTION_TIME=<s>
     (the line that run_all.sh extracts)

Run:  mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi2.py
"""
import heapq
import os
import sys
from collections import Counter

from mpi4py import MPI

VERSION = "mpi2"


def cargar_consulta(consulta_path, case_sensitive=False):
    """Reads consulta.txt and returns the set of target words (same as baseline)."""
    if not os.path.isfile(consulta_path):
        raise FileNotFoundError(f"Query file not found: {consulta_path}")

    with open(consulta_path, "r", encoding="utf-8") as f:
        palabras = [line.strip() for line in f if line.strip()]

    if not case_sensitive:
        palabras = [w.lower() for w in palabras]

    return set(palabras)


def listar_archivos(dataset_dir):
    """Returns the sorted list of file_*.txt names in dataset_dir.
    os.scandir gives the file type without an extra stat() per file, which
    matters on Docker Desktop bind mounts where every metadata call is slow."""
    with os.scandir(dataset_dir) as it:
        return sorted(
            e.name for e in it
            if e.name.startswith("file_") and e.name.endswith(".txt") and e.is_file()
        )


def obtener_tamanos(dataset_dir, archivos):
    """Returns [(bytes, name), ...] for the given files (metadata only)."""
    return [(os.path.getsize(os.path.join(dataset_dir, fname)), fname) for fname in archivos]


def repartir_archivos(con_tamano, size):
    """Size-aware distribution (LPT greedy): largest file first, always to the
    process with the fewest bytes assigned so far (min-heap)."""
    con_tamano = sorted(con_tamano, reverse=True)
    heap = [(0, r) for r in range(size)]          # (bytes assigned, rank)
    bloques = [[] for _ in range(size)]
    for nbytes, fname in con_tamano:
        carga, r = heapq.heappop(heap)
        bloques[r].append(fname)
        heapq.heappush(heap, (carga + nbytes, r))
    return bloques


def contar_palabras_en_archivos(dataset_dir, archivos, palabras_objetivo, case_sensitive=False):
    """Counts the target words in the given files.
    The inner loop is identical to contar_palabras_en_corpus() of the baseline."""
    freq_local = Counter()
    total_tokens_leidos = 0
    bytes_leidos = 0

    for fname in archivos:
        ruta = os.path.join(dataset_dir, fname)

        with open(ruta, "r", encoding="utf-8") as f:
            bytes_leidos += os.fstat(f.fileno()).st_size   # size of the open file, no extra path lookup
            for linea in f:
                palabras = linea.split()

                if not case_sensitive:
                    palabras = [w.lower() for w in palabras]

                total_tokens_leidos += len(palabras)

                for w in palabras:
                    if w in palabras_objetivo:
                        freq_local[w] += 1

    return freq_local, total_tokens_leidos, bytes_leidos


def comparar_con_baseline(resultados_path, freq_global):
    """True/False if baseline_results.csv exists, None otherwise."""
    if not os.path.isfile(resultados_path):
        return None
    referencia = {}
    with open(resultados_path, "r", encoding="utf-8") as f:
        next(f)  # header: palabra,conteo
        for linea in f:
            palabra, conteo = linea.rstrip("\n").rsplit(",", 1)
            referencia[palabra] = int(conteo)
    return referencia == dict(freq_global)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    consulta_name = "consulta.txt"
    case_sensitive = False
    top_n = 10
    baseline_file = "baseline_results.csv"

    comm.Barrier()                      # all processes start the clock together
    t0 = MPI.Wtime()

    # 1) and 3): rank 0 reads the query and lists the corpus
    palabras_objetivo, bloques = None, None
    if rank == 0:
        if not os.path.isdir(dataset_dir):
            print(f"Error: dataset folder not found: {dataset_dir}", flush=True)
            comm.Abort(1)
        try:
            palabras_objetivo = cargar_consulta(
                os.path.join(dataset_dir, consulta_name), case_sensitive=case_sensitive)
        except FileNotFoundError as e:
            print("Error:", e, flush=True)
            comm.Abort(1)
        archivos = listar_archivos(dataset_dir)
    else:
        archivos = None

    # 2) broadcast of the query words (and of the file names)
    palabras_objetivo = comm.bcast(palabras_objetivo, root=0)
    archivos = comm.bcast(archivos, root=0)

    # the 3000 file sizes are read in parallel: each process stats 1/p of the
    # names (round-robin) and rank 0 gathers them, so the extra metadata cost
    # of v2 is divided by p instead of being paid serially by rank 0
    tamanos_parciales = comm.gather(obtener_tamanos(dataset_dir, archivos[rank::size]), root=0)
    if rank == 0:
        bloques = repartir_archivos([t for parte in tamanos_parciales for t in parte], size)

    # 4) distribution of the files
    mis_archivos = comm.scatter(bloques, root=0)
    tiempo_setup = MPI.Wtime() - t0     # on rank 0: query read + listing + distribution

    # 5) local processing (local processing time t_i)
    t_local_inicio = MPI.Wtime()
    freq_local, tokens_locales, bytes_locales = contar_palabras_en_archivos(
        dataset_dir, mis_archivos, palabras_objetivo, case_sensitive=case_sensitive)
    tiempo_local = MPI.Wtime() - t_local_inicio

    estadisticas = (rank, len(mis_archivos), bytes_locales, tokens_locales, tiempo_local)

    # 6) gather the partial results in rank 0
    todas_freq = comm.gather(freq_local, root=0)
    todas_estadisticas = comm.gather(estadisticas, root=0)

    if rank == 0:
        # 7) global result
        freq_global = Counter()
        for parcial in todas_freq:
            freq_global.update(parcial)
        elapsed = MPI.Wtime() - t0

        tiempos = [e[4] for e in todas_estadisticas]
        media = sum(tiempos) / len(tiempos)
        li = max(tiempos) / media if media > 0 else float("nan")
        spread = (max(tiempos) - min(tiempos)) / media if media > 0 else float("nan")

        archivos_procesados = sum(e[1] for e in todas_estadisticas)
        total_tokens_leidos = sum(e[3] for e in todas_estadisticas)
        total_ocurrencias = sum(freq_global.values())

        print(f"=== {VERSION}: size-aware LPT distribution, p={size} ===")
        print("Per-process statistics:")
        for r, n_arch, nb, ntok, t in sorted(todas_estadisticas):
            print(f"PROC rank={r} files={n_arch} bytes={nb} tokens={ntok} local_time={t:.6f}")
        print(f"Files processed: {archivos_procesados}")
        print(f"Total tokens read: {total_tokens_leidos}")
        print(f"Total occurrences found: {total_ocurrencias}")
        print(f"LOAD_IMBALANCE={li:.4f}")
        print(f"RELATIVE_SPREAD={spread:.4f}")
        print(f"SETUP_TIME={tiempo_setup:.6f}")
        print(f"MAX_LOCAL_TIME={max(tiempos):.6f}")
        print(f"OVERHEAD_TIME={elapsed - max(tiempos):.6f}")
        print(f"\nTop {top_n} query words in the corpus:")
        for palabra, cuenta in freq_global.most_common(top_n):
            print(f"  {palabra}: {cuenta}")
        coincide = comparar_con_baseline(os.path.join(dataset_dir, baseline_file), freq_global)
        print(f"\nMATCHES_BASELINE={'n/a' if coincide is None else coincide}")
        print(f"EXECUTION_TIME={elapsed:.6f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
