# 1. Parallel Word Counting in a Text Corpus with MPI

> Lab 2 – Parallel and Distributed Computing. Author: Daniel Echeverria.

## 2. Problem Description

The dataset generator (`generator.py`, seed 42) creates `dataset/consulta.txt` (200 query words, one per line) and a corpus of 3000 files `dataset/file_XXXX.txt` (≈45 M tokens, Zipf-weighted vocabulary of 2000 words). The goal is to count how many times each query word appears in the corpus (case-insensitive, whitespace tokenization) and report the 10 most frequent ones.

The file sizes are deliberately heterogeneous:

| Category | Share of files | Words per file |
|---|---|---|
| Small | 60 % | 2 000 – 5 000 |
| Medium | 30 % | 10 000 – 20 000 |
| Large | 10 % | 50 000 – 100 000 |

The ~300 large files hold about half of all tokens, so giving each process the same **number** of files does not give it the same **amount of work**. This is the load-balancing problem studied in the lab.

## 3. Environment and Execution Instructions

### Environment

| Item | Value |
|---|---|
| Docker image | `augustosalazar/slim-mpi:2` (Python 3 + mpi4py + Open MPI) |
| Host | Laptop, plugged in during all runs, other heavy applications closed |
| Host CPU | Intel Core i7-11800H @ 2.30 GHz base: **8 physical cores / 16 logical processors** (Hyper-Threading), 24 MB L3 |
| Host RAM | 16 GB |
| OS | Windows 11 Home Single Language, version 26H2 (build 26300) |
| Docker | Docker Desktop, Docker version 29.8.2 (build 7fc2dff); the project folder is bind-mounted at `/app` |
| Process count used `p` | **16** (`detectar_procesos_mpi.py` → `N RECOMENDADO: 16`, median 1.132 s) |

### Repository files

| File | Description |
|---|---|
| `generator.py` | Provided, unchanged. Generates `dataset/`. |
| `baseline_secuencial.py` | Provided, unchanged. Sequential reference. |
| `run_all.sh` | Provided, unchanged. Generates the data, runs the baseline, mpi1 and mpi2 and stores logs in `results/logs/`. |
| `mpi1.py` | MPI version 1 (static block distribution). |
| `mpi2.py` | MPI version 2 (size-aware LPT distribution). |
| `detectar_procesos_mpi.py` | Provided by the course; picks the process count `p`. |
| `analyze.py` | Our helper: parses the logs and computes T_seq, T_p, S_p, E_p, LI, spread and the time breakdown → `results/metrics.md`. |
| `results/` | Final experiment: `logs/` (every run), `times_summary.csv` (written by `run_all.sh`), `metrics.md` (written by `analyze.py`). |
| `results_attempt1/` | Logs of the first attempt with an earlier version of the code (see section 4d). |
| `.gitattributes` / `.gitignore` | Keep LF line endings for the container; keep the generated `dataset/` out of the repository. |

### Commands

Run them from the repository folder. The commands are the same on every OS; only the volume flag changes:
Linux / macOS / Git Bash: `-v "$(pwd)":/app` · Windows CMD: `-v "%cd%:/app"` · Windows PowerShell: `-v "${PWD}:/app"`.

```bash
# 1) Choose p (prints "N RECOMENDADO: <p>" at the end)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py

# 2) Run the whole experiment with that p (3 runs per version). We used p = 16.
#    run_all.sh is unchanged: NP_LIST and RUNS are its own environment variables.
docker run --rm -e NP_LIST=16 -e RUNS=3 -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh

# 3) Two extra baseline runs (run_all.sh runs it only once) so T_seq is also an average of 3
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh -c "python /app/baseline_secuencial.py > /app/results/logs/baseline_run2.log"
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh -c "python /app/baseline_secuencial.py > /app/results/logs/baseline_run3.log"

# 4) Compute all metrics -> results/metrics.md
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/analyze.py
```

Note: `.gitattributes` forces LF line endings for `*.sh` and `*.py`, so `run_all.sh` still works inside the Linux container after cloning the repository on Windows.

A single run can also be launched by hand:

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 16 python /app/mpi1.py
```

### 3a. Experimental Plan

| Aspect | Decision |
|---|---|
| Hypothesis H1 | MPI v1 is faster than the sequential baseline, but S_p < p because of communication/start-up overhead and load imbalance. |
| Hypothesis H2 | Distributing files by size (v2) lowers LI toward 1.0 and therefore lowers T_p. |
| Independent variable | Distribution strategy (static block by file count vs. size-aware LPT). |
| Fixed factors | Same dataset (seed 42), same Docker image, same machine, same `p`, same counting code. |
| Dependent variables | Total time T_p, local time t_i per process, S_p, E_p, LI, relative spread. |
| Repetitions | 3 runs per configuration (baseline, mpi1, mpi2); results are averaged and the standard deviation is reported. |
| Correctness check | Every MPI run compares its full global count with `dataset/baseline_results.csv` (`MATCHES_BASELINE=True`) and prints the same top 10. |
| Conditions | Machine plugged in, other heavy applications closed, runs done back-to-back. |

### 3b. Sequential baseline (what it does and how)

`baseline_secuencial.py` reads `consulta.txt` into a set, iterates over every `file_*.txt`, splits each line on whitespace, lower-cases each token and increments a `Counter` when the token is in the query set. It measures the time of the counting phase with `time.perf_counter()`, saves the full result to `dataset/baseline_results.csv` and prints the top 10.

### 3c. MPI version 1 (what it does and how)

1. All ranks synchronize with `Barrier` and rank 0 starts the clock (`MPI.Wtime`).
2. Rank 0 reads `consulta.txt` and broadcasts the query set with `comm.bcast`.
3. Rank 0 lists `file_*.txt` with `os.scandir` (sorted by name) and splits it into `p` contiguous blocks of `⌊3000/p⌋` or `⌈3000/p⌉` files; the blocks are sent with `comm.scatter`.
4. Each rank counts its files with **exactly the same counting logic as the baseline** (same function and variable names: `cargar_consulta`, `palabras_objetivo`, `case_sensitive`, `top_n`, `dataset_dir`, `consulta_name`, `freq_global`, …) and measures its local processing time t_i.
5. Partial `Counter`s and per-process statistics (files, bytes, tokens, t_i) are collected with `comm.gather`.
6. Rank 0 merges the counters, prints one `PROC` line per rank, LI, spread, the top 10, `MATCHES_BASELINE` and `EXECUTION_TIME=` (the line that `run_all.sh` extracts).

### 3d. Test procedure

1. Generate the dataset and run the baseline (`run_all.sh` does both) → T_seq and reference output.
2. Run `mpi1.py` and `mpi2.py` 3 times each with the selected `p`.
3. Verify correctness: every log must contain `MATCHES_BASELINE=True` and the same top 10 as the baseline.
4. Run `analyze.py` to compute T_p (mean of 3), S_p = T_seq / T_p, E_p = S_p / p, LI = max(t_i) / mean(t_i) and spread = (max(t_i) − min(t_i)) / mean(t_i) for each run, then average.

## 4. Experimental Plan Execution

All numbers below come from `results/metrics.md` and `results/logs/` (p = 16, 3 runs per configuration). All 6 MPI runs printed `MATCHES_BASELINE=True`: the 200 counts are identical to `baseline_results.csv`, and the top 10 is identical to the baseline's.

### 4a. Sequential baseline timing

| Run | Time (s) |
|---|---|
| 1 | 16.6301 |
| 2 | 15.7824 |
| 3 | 17.4352 |
| **Mean T_seq** | **16.6159** (std 0.8265) |

3000 files, 44 951 458 tokens (275.05 MB), 3 631 778 occurrences. Sequential throughput: 275.05 MB / 16.62 s = **16.6 MB/s**.

Top 10 (baseline): a 785774, para 392156, sus 228913, otros 105530, ante 99832, unos 88794, otra 83901, vosotros 61617, mios 58420, tuya 56635.

### 4b. MPI version 1 timing results

| Run | Total time (s) | Setup (s) | max(t_i) (s) | Overhead = total − max(t_i) (s) | LI | Spread | Matches baseline |
|---|---|---|---|---|---|---|---|
| 1 | 3.3452 | 0.0621 | 3.2820 | 0.0632 | 1.1253 | 0.2643 | True |
| 2 | 3.3946 | 0.0273 | 3.3660 | 0.0286 | 1.1258 | 0.2438 | True |
| 3 | 3.4628 | 0.0325 | 3.4293 | 0.0335 | 1.1141 | 0.2343 | True |

**T_p = 3.4009 s** (std 0.0590) → **S_p = 16.6159 / 3.4009 = 4.886**, **E_p = 4.886 / 16 = 0.305**.

### 4c. Load imbalance evidence (per-process local times and LI)

MPI v1, per-process local processing time t_i (s):

| Rank | Files | MB | Tokens | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|---|---|---|
| 0 | 188 | 17.59 | 2875334 | 2.9963 | 3.0724 | 3.1718 | 3.0802 |
| 1 | 188 | 19.17 | 3131769 | 3.1249 | 3.1815 | 3.2803 | 3.1956 |
| 2 | 188 | 15.80 | 2584424 | 2.8242 | 2.8704 | 2.9421 | 2.8789 |
| 3 | 188 | 13.58 | 2219696 | 2.5112 | 2.6371 | 2.7270 | 2.6251 |
| **4** | 188 | **22.19** | 3626652 | 3.2820 | 3.3660 | 3.4293 | **3.3591** |
| 5 | 188 | 15.65 | 2556891 | 2.7877 | 2.8441 | 2.9688 | 2.8669 |
| 6 | 188 | 18.02 | 2944097 | 3.0195 | 3.0927 | 3.1976 | 3.1033 |
| 7 | 188 | 18.64 | 3048154 | 3.0854 | 3.1454 | 3.2323 | 3.1544 |
| 8 | 187 | 15.04 | 2457345 | 2.6961 | 2.7946 | 2.8361 | 2.7756 |
| 9 | 187 | 17.65 | 2883674 | 3.0327 | 3.0580 | 3.1951 | 3.0953 |
| 10 | 187 | 16.25 | 2656051 | 2.8128 | 2.9080 | 2.9883 | 2.9030 |
| 11 | 187 | 17.21 | 2812767 | 2.9552 | 3.0028 | 3.0965 | 3.0182 |
| 12 | 187 | 19.56 | 3196393 | 3.1144 | 3.1844 | 3.2717 | 3.1902 |
| **13** | 187 | **13.54** | 2212406 | 2.5169 | 2.6479 | 2.7082 | **2.6243** |
| 14 | 187 | 15.50 | 2532423 | 2.7736 | 2.8358 | 2.9172 | 2.8422 |
| 15 | 187 | 19.66 | 3213382 | 3.1311 | 3.1986 | 3.2851 | 3.2049 |

- Every process receives 187–188 files, but the bytes range from 13.54 MB (rank 13) to 22.19 MB (rank 4), a ratio of 1.64.
- The slowest process in every run is rank 4, which has the most bytes. The fastest are ranks 13 and 3, which have the fewest.
- **Mean LI = 1.122** and **mean spread = 0.247**: the slowest process takes 12 % longer than the average one, and the fastest finishes about 25 % of the mean time earlier than the slowest.

Because the dataset is deterministic (seed 42), the bytes each rank receives can be computed exactly for any p. Byte-load imbalance = max(bytes_i) / mean(bytes_i):

| p | v1 static blocks | v2 size-aware LPT |
|---|---|---|
| 2 | 1.022 | 1.000 |
| 4 | 1.082 | 1.000 |
| 8 | 1.100 | 1.000 |
| 12 | 1.179 | 1.000 |
| **16** | **1.291** | **1.000** |

At p = 16 the work given to rank 4 is 29 % above the mean. The measured time LI (1.12) is lower than the byte LI (1.29). The heavy processes counted at 6.5–6.8 MB/s and the light ones at 5.0–5.4 MB/s. When the light processes finish, the heavy ones get more of the CPU (see 5b).

### 4d. MPI version 2: how it corrects the imbalance, and its results (total execution time, speedup, efficiency, LI)

**How:** `mpi2.py` counts exactly like v1; only the distribution changes.
- Rank 0 lists the files and broadcasts the names.
- Each process reads the sizes of 1/p of the files with `os.path.getsize` (metadata only, no content is read), and rank 0 gathers them.
- Rank 0 sorts the files from largest to smallest and assigns each one to the rank that currently has the fewest bytes (greedy *Longest Processing Time first*, using a min-heap).

The assignment is still static, decided once before the work. Ranks receive 186–189 files but 17.18–17.20 MB each (byte LI = 1.0004).

**First attempt (`results_attempt1/`).** The first version had two differences from the final one:
- Rank 0 read all 3000 sizes by itself.
- Both versions listed the files with `os.listdir` + `os.path.isfile`, one extra metadata call per file.

Inside Docker Desktop on Windows, every metadata call on the bind-mounted `/app` folder is slow. The time outside the counting (total − max(t_i)) was 4.22 s for v1 and 7.57 s for v2. As a result, T_p was 7.49 s for v1 and **11.07 s** for v2, so v2 was 48 % slower than v1. The 3.35 s difference matches 3000 serial `getsize` calls at ≈1.1 ms each. We therefore made two changes:
- Both versions now list with `os.scandir` (file type without an extra call).
- In v2 the size lookup is split among the processes.

These changes cut the v1 overhead from 4.22 s to 0.04 s and the v2 overhead from 7.57 s to 0.39 s. Everything else in this report uses the final code.

**Results (final code):**

| Run | Total time (s) | Setup (s) | max(t_i) (s) | Overhead (s) | LI | Spread | Matches baseline |
|---|---|---|---|---|---|---|---|
| 1 | 3.8212 | 0.4058 | 3.4132 | 0.4079 | 1.0099 | 0.0242 | True |
| 2 | 3.8280 | 0.3981 | 3.4282 | 0.3998 | 1.0317 | 0.0624 | True |
| 3 | 3.7099 | 0.3484 | 3.3596 | 0.3503 | 1.0152 | 0.0325 | True |

Per-process mean t_i in v2 goes from 3.3037 s to 3.3717 s for all 16 ranks (full table in `results/metrics.md`).

**Comparison (p = 16, T_seq = 16.6159 s):**

| Version | T_p mean (s) | T_p std (s) | Speedup S_p | Efficiency E_p | LI | Spread | mean max(t_i) (s) | mean overhead (s) |
|---|---|---|---|---|---|---|---|---|
| mpi1 (static blocks) | **3.4009** | 0.0590 | **4.886** | **0.305** | 1.1217 | 0.2474 | 3.3591 | 0.0418 |
| mpi2 (size-aware LPT) | 3.7864 | 0.0663 | 4.388 | 0.274 | **1.0190** | **0.0397** | 3.4004 | 0.3860 |

## 5. Analysis

### 5a. Parallel performance

*Did MPI version 1 improve on the sequential baseline, and how close was its speedup to the ideal (S_p = p)? What explains the gap?*

Yes. MPI v1 reduced the time from 16.62 s to 3.40 s, a speedup of **4.89**. However, this is far from the ideal S_p = 16 (efficiency **0.305**). The time breakdown shows where the gap is:

- **The overhead is not the cause.** T_p = 3.40 s ≈ 0.04 s outside the counting + 3.36 s for the slowest process. Reading the query, listing, `bcast`, `scatter`, `gather` and merging take about 1 % of the time.
- **The imbalance explains only a small part.** With perfect balance the slowest process would take about the mean time instead of LI × mean, so at most ≈11 % of T_p (1 − 1/1.12) could be saved.
- **The main cause is that each token costs much more when 16 processes run at once.** The baseline needs 16.6 s of work. In v1 the 16 local times add up to 47.9 s on average, 2.9× more for the same 45 M tokens. The machine as a whole processes 275 MB / 3.36 s ≈ 82 MB/s, which is only 4.9× the sequential 16.6 MB/s. Three reasons explain this:
  - The CPU has **8 physical cores**: the 16 logical processors are Hyper-Threading pairs that share the execution units of one core, so p = 16 does not mean 16 full cores. Measured against the 8 physical cores, the efficiency is 4.89 / 8 = 0.61.
  - A laptop CPU runs at a lower clock speed when all cores are busy than when only one is.
  - All processes share the 24 MB L3 cache and the memory bandwidth, and run inside Docker Desktop's virtual machine.

  `detectar_procesos_mpi.py` chose N = 16 for a pure-CPU prime-counting task, which hides these shared-resource limits.

### 5b. Load balance

*What evidence of load imbalance did version 1 show (per-process local times and LI), and did version 2 reduce it enough to produce a real performance improvement?*

**Evidence in v1.** The local times follow the bytes received. Rank 4 (22.19 MB) was the slowest in all 3 runs (mean 3.36 s). Ranks 13 and 3 (≈13.6 MB) were the fastest (≈2.62 s). The results were mean LI = 1.122 and spread = 0.247, caused by a byte imbalance of 1.29 even though the file counts were equal (187–188).

**v2 reduced the imbalance indicator but not the time.** LI dropped from 1.122 to **1.019** and the spread from 0.247 to **0.040**. All ranks ended within 3.30–3.37 s. However, the slowest process did not finish earlier: mean max(t_i) was 3.40 s in v2 versus 3.36 s in v1. The reason is that the CPU was already saturated:
- The aggregate throughput was the same in both versions: ≈82 MB/s in v1 and ≈81 MB/s in v2.
- In v1, when the light ranks finish, the heavy ranks get the freed hardware threads and speed up (6.5–6.8 MB/s versus 5.0–5.4 MB/s). This is why the time LI (1.12) is much lower than the byte LI (1.29).
- In v2 all 16 processes compete until the end, each at about 5.1 MB/s.

With 16 processes on 8 cores, the finishing time is set mainly by *total work / machine throughput*, and that does not change with the distribution.

**Net result.** v2 adds 0.35 s of setup (0.384 s versus 0.041 s) to look up the sizes through the Docker bind mount. Its T_p is **3.79 s ± 0.07 versus 3.40 s ± 0.06**: v2 is 11 % slower, and the difference (0.39 s) is about 6 times the run-to-run standard deviation, so it is real and not noise. **Version 2 did not produce a real performance improvement in this environment.** Our hypothesis H2 is rejected: a better LI did not lower T_p. The first attempt (T_p = 11.07 s) shows that, here, the cost of collecting the information needed for balancing can be larger than the time lost to imbalance.

### 5c. Limitations

*What limitations affected your experiment, and how much do they weaken your conclusions?*

| Limitation | Effect on the conclusions |
|---|---|
| Only 3 runs per configuration on one laptop. | Low for the v1-vs-v2 comparison: the 0.39 s difference is about 6× the standard deviation (0.06–0.07 s). |
| Baseline variability (std 0.83 s, 5 %). | S_p and E_p carry about ±5 % uncertainty. It does not affect which version is better, because both use the same T_seq. |
| Docker Desktop on Windows: the code runs in a virtual machine, and metadata calls on the bind-mounted folder cost ≈1 ms each. | Medium. The setup cost of v2 (0.35 s) is specific to this environment. On native Linux or a cluster with local disks it would be much smaller, and v2 could tie with or beat v1. The conclusion "v2 is slower" is valid for this setup, not in general. |
| Only p = 16 was evaluated (as the lab requires), and p = 16 is more than the 8 physical cores. | Medium. The finding that balancing does not help depends on the CPU being saturated. With p ≤ 8 each process would have its own core, so imbalance would leave cores idle and v2 would likely help. This was not measured. |
| `run_all.sh` runs the three mpi1 runs and then the three mpi2 runs. | Low. A warmer CPU could slightly penalize mpi2. However, most of v2's extra time is setup, which is metadata access rather than computation, and max(t_i) differs by only 0.04 s. |
| The 275 MB dataset fits in RAM, so after the first read the files come from the OS cache. | It makes the experiment measure CPU throughput rather than disk speed. This is the same for both versions, so it does not bias the comparison. |
| File size is used as a proxy for work. | Valid here: time follows bytes (rank 4 is always the slowest). It would not hold if the cost per byte varied between files. |
| The first attempt used an earlier version of the code. | Its numbers are only used to show the cost of serial metadata calls, not for the final comparison. |

## 6. Conclusions

*In one paragraph, an evidence-based judgment of which distribution strategy we recommend for this problem and why, citing the measured speedup, efficiency, and LI.*

For this problem, on this machine (i7-11800H, 8 cores / 16 threads, Docker Desktop, p = 16), we recommend the **static block distribution of MPI version 1**. It achieved the best measured performance: T_p = 3.40 s, S_p = 4.89 and E_p = 0.305, against T_p = 3.79 s, S_p = 4.39 and E_p = 0.274 for the size-aware LPT distribution of version 2. The 0.39 s difference is about 6 times the run-to-run standard deviation. Version 2 did its intended job on the balance indicator: LI went from 1.12 to 1.02 and the spread from 0.25 to 0.04. However, this did not shorten the slowest process (3.36 s versus 3.40 s). With 16 processes on 8 physical cores the CPU is saturated, so in v1 the heavy processes simply take over the hardware threads freed by the light ones, and the total throughput stays at ≈82 MB/s in both versions. Version 2 also has to pay 0.35 s to read 3000 file sizes through the Docker bind mount, and 3.35 s in our first attempt. The gap to the ideal speedup is caused by shared hardware (Hyper-Threading, all-core clock speed, cache and memory bandwidth), not by the distribution strategy. Size-aware balancing would only be worth its cost when processes do not share cores (p ≤ number of physical cores, or a real cluster) and file sizes can be obtained cheaply. Neither condition holds in this environment.
