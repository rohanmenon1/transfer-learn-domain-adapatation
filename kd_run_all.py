"""Runs every KD / ERM job in order, resumable:  python kd_run_all.py [--smoke] [--dry-run]

- The job list is rebuilt each round; a job is done when its result JSON is marked complete, so the runner can
  be stopped and restarted at any time.
- A failed job is retried (up to MAX_ATTEMPTS), then skipped together with the jobs that depend on it.
- Each job logs to logs/kd/<id>.log; logs/kd/STATUS.txt gets one line per job with an ETA.
- After every job, kd_report.py rebuilds kd_results/RESULTS.md and the figures.
- Pauses while free disk space is below 5 GB.
"""
import argparse
import ctypes
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
TARGETS = ["sketch", "quickdraw", "painting", "infograph", "clipart", "real"]
# expected minutes per job kind on an RTX 3080 Laptop GPU (8 GB); rescaled by measured times at run time
EST = {"distance": 5, "expert": 25, "src-teacher": 30, "src-student": 20, "lodo-teacher": 60,
       "lodo-student": 45, "adapt": 6, "adapt12": 11}
TEACHER_ACCUM = 2  # ViT-B teacher: batch 64 as 2 x 32 to fit in 8 GB
MAX_ATTEMPTS = 3   # transient out-of-memory errors usually succeed on a retry


def jobs_list(smoke, res_dir, models_dir):
    targets = ["sketch"] if smoke else TARGETS
    sources = ["sketch"] if smoke else DOMAINS
    J = []

    def add(id_, kind, args, result, needs=(), script="kd_train.py"):
        sub = [kind] if script == "kd_train.py" else []
        J.append({"id": id_, "kind": kind, "cmd": [PY, script, *sub, *args],
                  "result": res_dir / result, "needs": list(needs)})

    def adapt(short, init, t, needs, lpft=False):
        # seed 0, then seeds 1-2 in a second file (same starting model, new 10-shot samples)
        base = ["--init", init, "--name", short, "--target", t] + (["--lpft"] if lpft else [])
        add(f"adapt2_{short}_to-{t}", "adapt", base + ["--seeds", "0"], f"adapt_{short}_to-{t}_k10_v2.json",
            needs, script="kd_adapt.py")
        add(f"adapt2s12_{short}_to-{t}", "adapt12", base + ["--seeds", "1", "2", "--tag", "v2_s12"],
            f"adapt_{short}_to-{t}_k10_v2_s12.json", needs, script="kd_adapt.py")

    def lodo_student(method, teacher, t, needs):
        n = f"{'erm' if method == 'erm' else f'kd-{method}-{teacher}'}_dinov3_lodo-{t}_5k_s0"
        add(n, "lodo-student", ["--method", method, "--teacher", teacher, "--target", t], n + ".json", needs)
        return n

    ck = lambda n: str(models_dir / f"{n}.pt")
    # 1. domain distances and single-domain ERM models (finetune.py recipe)
    add("distance_dinov3", "distance", [], "distance_dinov3.json", script="kd_distance.py")
    for d in DOMAINS:
        add(f"expert_dinov3_{d}_s0", "expert", ["--domain", d], f"expert_dinov3_{d}_s0.json")
    # 2. train on one domain, adapt to each of the other five: ERM and KD (ViT-B teacher)
    for s_ in sources:
        tn, sn = f"teacher-vitb_dinov3_src-{s_}_s0", f"kd-kd-vitb_dinov3_src-{s_}_s0"
        add(tn, "src-teacher", ["--domain", s_, "--accum", str(TEACHER_ACCUM)], tn + ".json")
        add(sn, "src-student", ["--domain", s_], sn + ".json", [tn])
    for s_ in sources:
        ex, sn = f"expert_dinov3_{s_}_s0", f"kd-kd-vitb_dinov3_src-{s_}_s0"
        for t in DOMAINS:
            if t != s_:
                adapt(f"expert-{s_}", ck(ex), t, [ex])
                adapt(f"kd-vitb-src-{s_}", ck(sn), t, [sn])
    # 3. baseline without DomainNet training: pretrained DINOv3, linear probe then fine-tune (LP-FT)
    for t in targets:
        adapt("dinov3-lpft", "dinov3", t, [], lpft=True)
    # 4. leave one domain out (train on the other five)
    for t in targets:
        erm = lodo_student("erm", "none", t, [])
        adapt("erm-5k", ck(erm), t, [erm])
        n = lodo_student("kd", "experts", t, [f"expert_dinov3_{d}_s0" for d in DOMAINS if d != t])
        adapt("kd-experts-5k", ck(n), t, [n])
        tn = f"teacher-vitb_dinov3_lodo-{t}_5k_s0"
        add(tn, "lodo-teacher", ["--target", t, "--accum", str(TEACHER_ACCUM)], tn + ".json")
        n = lodo_student("kd", "vitb", t, [tn])
        adapt("kd-vitb-5k", ck(n), t, [n])
    return J


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def done(j):
    return load(j["result"]).get("complete") is True


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def pid_alive(pid):
    """True if pid is a running Python process (Windows only; elsewhere assume not running)."""
    try:
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        code, buf, n = ctypes.c_ulong(), ctypes.create_unicode_buffer(512), ctypes.c_ulong(512)
        alive = h and k32.GetExitCodeProcess(h, ctypes.byref(code)) and code.value == 259
        return bool(alive and k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) and "python" in buf.value.lower())
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test, outputs in logs/kd_smoke/")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs that would run, then exit")
    a = ap.parse_args()
    base = REPO / "logs" / "kd_smoke" if a.smoke else REPO
    res_dir, models_dir = base / "results", base / "models"
    logs = REPO / "logs" / ("kd_smoke" if a.smoke else "kd")
    for p in (logs, res_dir, models_dir):
        p.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        J = jobs_list(a.smoke, res_dir, models_dir)
        todo = [j for j in J if not done(j)]
        print(f"{len(J)} jobs, {len(todo)} to run")
        for j in todo:
            print(f"  {j['id']:<55} {' '.join(j['cmd'][1:3])}")
        return
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if a.smoke:
        env["KD_SMOKE"] = "1"
    try:  # keep the machine awake while jobs run (Windows: ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    lock = logs / "runner.pid"  # one runner at a time
    try:
        old = int(lock.read_text())
        if old != os.getpid() and pid_alive(old):
            print(f"another runner (pid {old}) is running; exiting")
            return
    except Exception:
        pass
    lock.write_text(str(os.getpid()))
    state_p = logs / "state.json"
    st = load(state_p) or {"attempts": {}, "sec": {}}
    status = logs / "STATUS.txt"

    def status_line(msg):
        with open(status, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%m-%d %H:%M} {msg}\n")

    status_line(f"runner started (pid {os.getpid()}{', SMOKE' if a.smoke else ''})")
    while True:
        J = jobs_list(a.smoke, res_dir, models_dir)
        ok = {j["id"] for j in J if done(j)}
        dead = {j["id"] for j in J if st["attempts"].get(j["id"], 0) >= MAX_ATTEMPTS and j["id"] not in ok}
        changed = True
        while changed:  # anything that needs a failed job is skipped too
            changed = False
            for j in J:
                if j["id"] not in dead and j["id"] not in ok and any(n in dead for n in j["needs"]):
                    dead.add(j["id"])
                    changed = True
        todo = [j for j in J if j["id"] not in ok and j["id"] not in dead]
        ready = [j for j in todo if all(n in ok for n in j["needs"])]
        if not ready:
            status_line(f"ALL DONE: {len(ok)} done, {len(dead)} failed/skipped: {', '.join(sorted(dead)) or 'none'}")
            break
        if shutil.disk_usage(str(REPO)).free / 1e9 < 5:
            status_line("PAUSED: less than 5 GB free on disk; re-checking every 10 min")
            while shutil.disk_usage(str(REPO)).free / 1e9 < 5:
                time.sleep(600)
            status_line("resumed: disk space is back")
            continue
        j = ready[0]
        fin = [(EST[k], s) for k, s in ((st.get("kind", {}).get(i), s) for i, s in st["sec"].items()) if k in EST]
        scale = (sum(s for _, s in fin) / 60 / sum(e for e, _ in fin)) if fin else 1.0
        rem = scale * sum(EST[x["kind"]] for x in todo) * 60
        status_line(f"start {j['id']} | {len(ok)}/{len(J)} done | all remaining ~{fmt(rem)}"
                    f" (finish ~{dt.datetime.now() + dt.timedelta(seconds=rem):%a %H:%M})")
        t0 = time.time()
        with open(logs / f"{j['id']}.log", "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} attempt {st['attempts'].get(j['id'], 0) + 1}\n"
                     f"{' '.join(j['cmd'])}\n")
            lf.flush()
            rc = subprocess.call(j["cmd"], cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
        sec = time.time() - t0
        if rc == 0 and done(j):
            st["sec"][j["id"]] = round(sec)
            st.setdefault("kind", {})[j["id"]] = j["kind"]
            r = load(j["result"])
            extra = {"expert": lambda: f"in-domain {r.get('in_domain')}% (ceiling {r.get('ceiling')})",
                     "src-teacher": lambda: f"in-domain {r.get('in_domain')}%, mean on other 5 {r.get('cross_mean')}%",
                     "src-student": lambda: f"in-domain {r.get('in_domain')}%, mean on other 5 {r.get('cross_mean')}%",
                     "lodo-student": lambda: f"held-out {r.get('heldout')}%",
                     "lodo-teacher": lambda: f"teacher held-out {r.get('heldout')}%",
                     "adapt": lambda: f"start {r.get('start_mean')}%, AUC {r.get('auc_mean')}",
                     }.get(j["kind"], lambda: "")()
            status_line(f"  done {j['id']} in {fmt(sec)} | {extra}")
        else:
            st["attempts"][j["id"]] = st["attempts"].get(j["id"], 0) + 1
            status_line(f"  FAILED {j['id']} (rc {rc}, attempt {st['attempts'][j['id']]}) after {fmt(sec)}; "
                        f"see logs/kd/{j['id']}.log")
        state_p.write_text(json.dumps(st, indent=1))
        try:
            subprocess.call([PY, "kd_report.py"] + (["--smoke"] if a.smoke else []), cwd=str(REPO), env=env,
                            stdout=open(logs / "report.log", "a"), stderr=subprocess.STDOUT, timeout=600)
        except Exception as e:
            status_line(f"  report failed: {e}")


if __name__ == "__main__":
    main()
