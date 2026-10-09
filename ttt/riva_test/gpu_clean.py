"""
GPU / CUDA cleaner: shows what holds GPU memory and frees it.

Two uses:
  1. From the shell, before or after a run (leftover processes from crashed runs keep VRAM busy):
       python gpu_clean.py                  # report: GPUs, memory, processes using them
       python gpu_clean.py --kill           # stop YOUR processes that hold GPU memory (asks first)
       python gpu_clean.py --kill --yes     # same, no question
       python gpu_clean.py --kill --gpu 0   # only processes on GPU 0
  2. Inside a script, to release memory the current process holds:
       from gpu_clean import free_gpu
       del model; free_gpu()

Only processes owned by you are stopped (SIGTERM, then SIGKILL after --timeout seconds).
Uses nvidia-smi; torch is optional (only for free_gpu).
"""

import argparse
import gc
import os
import signal
import subprocess
import sys
import time


def free_gpu(verbose=True):
    """Release cached CUDA memory held by THIS process (call after `del model`)."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    if verbose:
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: allocated {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB, "
                  f"reserved {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB (this process)")


def smi(query, kind):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("nvidia-smi not found: no NVIDIA driver on this machine.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"nvidia-smi failed: {e.stderr.strip()}")
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpus():
    rows = smi("index,uuid,name,memory.used,memory.total,utilization.gpu", "gpu")
    return [{"index": int(r[0]), "uuid": r[1], "name": r[2], "used": int(r[3]), "total": int(r[4]),
             "util": r[5]} for r in rows]


def proc_info(pid):
    """(owner uid, command line) or (None, None) if the pid is not visible (e.g. other container)."""
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        return uid, cmd
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None, None


def gpu_procs(uuid_to_index):
    procs = []
    for r in smi("pid,used_memory,gpu_uuid", "compute-apps"):
        pid = int(r[0])
        uid, cmd = proc_info(pid)
        procs.append({"pid": pid, "mem": int(r[1]) if r[1].isdigit() else 0,
                      "gpu": uuid_to_index.get(r[2], "?"), "uid": uid, "cmd": cmd})
    return procs


def report():
    gs = gpus()
    print("GPUs:")
    for g in gs:
        print(f"  GPU {g['index']}: {g['name']}  {g['used']:,} / {g['total']:,} MiB used  util {g['util']}%")
    procs = gpu_procs({g["uuid"]: g["index"] for g in gs})
    print("Processes using GPU memory:" if procs else "No processes are using GPU memory.")
    me = os.getuid()
    for p in procs:
        owner = "you" if p["uid"] == me else ("?" if p["uid"] is None else f"uid {p['uid']}")
        print(f"  GPU {p['gpu']}  pid {p['pid']:>7}  {p['mem']:>7,} MiB  [{owner}]  {(p['cmd'] or '(not visible)')[:90]}")
    return gs, procs


def kill(procs, timeout):
    for p in procs:
        try:
            os.kill(p["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + timeout
    alive = list(procs)
    while alive and time.time() < deadline:
        time.sleep(0.5)
        alive = [p for p in alive if os.path.exists(f"/proc/{p['pid']}")]
    for p in alive:
        print(f"  pid {p['pid']} still running -> SIGKILL")
        try:
            os.kill(p["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Show and free GPU memory.")
    ap.add_argument("--kill", action="store_true", help="stop your processes that hold GPU memory")
    ap.add_argument("--gpu", type=int, nargs="*", help="only these GPU indexes")
    ap.add_argument("--yes", "-y", action="store_true", help="don't ask before stopping")
    ap.add_argument("--timeout", type=float, default=10, help="seconds between SIGTERM and SIGKILL")
    args = ap.parse_args()

    _, procs = report()
    if not args.kill:
        return
    me = os.getuid()
    targets = [p for p in procs if p["uid"] == me and p["pid"] != os.getpid()
               and (args.gpu is None or p["gpu"] in args.gpu)]
    others = [p for p in procs if p["uid"] not in (me,)]
    if others:
        print(f"\nSkipping {len(others)} process(es) not owned by you (or not visible from here).")
    if not targets:
        print("Nothing of yours to stop.")
        return
    print(f"\nWill stop {len(targets)} process(es): " + ", ".join(str(p["pid"]) for p in targets))
    if not args.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return
    kill(targets, args.timeout)
    time.sleep(1)
    print()
    report()


if __name__ == "__main__":
    main()
