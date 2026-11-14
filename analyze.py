"""
Computes the lab metrics from the logs written by run_all.sh.

Reads:   results/logs/baseline*.log, results/logs/mpi1_np*_run*.log, results/logs/mpi2_np*_run*.log
Writes:  results/metrics.md  (markdown tables ready to paste into README.md)

Metrics:
  T_seq = mean of all baseline runs found
  T_p   = mean EXECUTION_TIME over runs
  S_p   = T_seq / T_p          E_p = S_p / p
  LI    = max(t_i) / mean(t_i) (t_i = local processing time of rank i)
  spread = (max(t_i) - min(t_i)) / mean(t_i)

Run:  docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/analyze.py
"""
import glob
import os
import re
import statistics as st
from collections import defaultdict

APP = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(APP, "results", "logs")
OUT = os.path.join(APP, "results", "metrics.md")

RE_SEQ = re.compile(r"Tiempo de ejecuci\S+n:\s*([\d.]+)")
RE_EXEC = re.compile(r"EXECUTION_TIME=([\d.]+)")
RE_PROC = re.compile(r"PROC rank=(\d+) files=(\d+) bytes=(\d+) tokens=(\d+) local_time=([\d.]+)")
RE_MATCH = re.compile(r"MATCHES_BASELINE=(\S+)")
RE_TOP = re.compile(r"^\s{2}(\S+): (\d+)$", re.M)
RE_SETUP = re.compile(r"SETUP_TIME=([\d.]+)")
RE_OVER = re.compile(r"OVERHEAD_TIME=([\d.]+)")
RE_NAME = re.compile(r"(mpi\d)_np(\d+)_run(\d+)\.log$")


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def main():
    lines = []
    out = lines.append

    # ---------- baseline ----------
    seq_times, seq_top = [], None
    for path in sorted(glob.glob(os.path.join(LOGS, "baseline*.log"))):
        txt = read(path)
        m = RE_SEQ.search(txt) or RE_EXEC.search(txt)
        if m:
            seq_times.append(float(m.group(1)))
            seq_top = seq_top or RE_TOP.findall(txt)
    if not seq_times:
        print("No baseline log found in", LOGS)
        return
    t_seq = st.mean(seq_times)

    out("## Sequential baseline\n")
    out("| Run | Time (s) |\n|---|---|")
    for i, t in enumerate(seq_times, 1):
        out(f"| {i} | {t:.4f} |")
    sd = f" (std {st.stdev(seq_times):.4f} s)" if len(seq_times) > 1 else ""
    out(f"\n**T_seq = {t_seq:.4f} s**{sd}\n")
    if seq_top:
        out("Top 10 (baseline): " + ", ".join(f"{w}={c}" for w, c in seq_top) + "\n")

    # ---------- MPI ----------
    runs = defaultdict(list)          # (prog, np) -> list of run dicts
    for path in glob.glob(os.path.join(LOGS, "mpi*_np*_run*.log")):
        m = RE_NAME.search(path)
        if not m:
            continue
        prog, np_, run = m.group(1), int(m.group(2)), int(m.group(3))
        txt = read(path)
        e = RE_EXEC.search(txt)
        procs = [tuple(map(float, p)) for p in RE_PROC.findall(txt)]
        if not e or not procs:
            print("WARNING: incomplete log", path)
            continue
        t_i = [p[4] for p in procs]
        mean_t = st.mean(t_i)
        mm = RE_MATCH.search(txt)
        ms, mo = RE_SETUP.search(txt), RE_OVER.search(txt)
        runs[(prog, np_)].append({
            "run": run, "total": float(e.group(1)), "procs": sorted(procs),
            "li": max(t_i) / mean_t, "spread": (max(t_i) - min(t_i)) / mean_t,
            "match": mm.group(1) if mm else "?",
            "setup": float(ms.group(1)) if ms else float("nan"),
            "maxloc": max(t_i),
            "over": float(mo.group(1)) if mo else float(e.group(1)) - max(t_i),
            "top": RE_TOP.findall(txt),
        })

    summary = []
    for (prog, p) in sorted(runs):
        rs = sorted(runs[(prog, p)], key=lambda r: r["run"])
        out(f"## {prog} - p = {p}\n")
        out("| Run | Total time (s) | Setup (s) | max(t_i) (s) | Overhead = total - max(t_i) (s) | LI | Spread | Matches baseline |")
        out("|---|---|---|---|---|---|---|---|")
        for r in rs:
            out(f"| {r['run']} | {r['total']:.4f} | {r['setup']:.4f} | {r['maxloc']:.4f} | {r['over']:.4f} | "
                f"{r['li']:.4f} | {r['spread']:.4f} | {r['match']} |")
        out("")
        out("Per-process local times t_i (s):\n")
        out("| Rank | Files | MB | Tokens | " + " | ".join(f"Run {r['run']}" for r in rs) + " | Mean |")
        out("|---" * (5 + len(rs)) + "|")
        for k in range(p):
            ref = rs[0]["procs"][k]
            ts = [r["procs"][k][4] for r in rs]
            out(f"| {int(ref[0])} | {int(ref[1])} | {ref[2] / 1e6:.2f} | {int(ref[3])} | "
                + " | ".join(f"{t:.4f}" for t in ts) + f" | {st.mean(ts):.4f} |")
        t_p = st.mean(r["total"] for r in rs)
        sd = st.stdev([r["total"] for r in rs]) if len(rs) > 1 else 0.0
        li = st.mean(r["li"] for r in rs)
        spread = st.mean(r["spread"] for r in rs)
        s_p = t_seq / t_p
        e_p = s_p / p
        top_ok = all(r["top"] == seq_top for r in rs) if seq_top else None
        out(f"\nT_p = {t_p:.4f} s (std {sd:.4f}), S_p = {s_p:.3f}, E_p = {e_p:.3f}, "
            f"mean LI = {li:.4f}, mean spread = {spread:.4f}, top-10 equals baseline: {top_ok}\n")
        over = st.mean(r["over"] for r in rs)
        maxloc = st.mean(r["maxloc"] for r in rs)
        summary.append((prog, p, len(rs), t_p, sd, s_p, e_p, li, spread, maxloc, over))

    out("## Summary\n")
    out(f"T_seq = {t_seq:.4f} s\n")
    out("| Version | p | Runs | T_p mean (s) | T_p std (s) | Speedup S_p | Efficiency E_p | LI | Spread | mean max(t_i) (s) | mean overhead (s) |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for prog, p, n, t_p, sd, s_p, e_p, li, spread, maxloc, over in summary:
        out(f"| {prog} | {p} | {n} | {t_p:.4f} | {sd:.4f} | {s_p:.3f} | {e_p:.3f} | {li:.4f} | {spread:.4f} | {maxloc:.4f} | {over:.4f} |")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
