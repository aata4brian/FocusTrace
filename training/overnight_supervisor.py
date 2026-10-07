"""Detached, deadline-limited experiment queue for one authorized device."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/metrics/overnight_20261006_v1"
CUTOFF = "2026-10-07T15:00:00+07:00"


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def timestamp():
    return datetime.now().astimezone().isoformat()


def jobs(role):
    if role == "laptop":
        return [
            dict(name="neural_binary_5fold_2s", suite="neural", target="binary", folds=5, seconds=2, weight=18),
            dict(name="neural_binary_5fold_6s", suite="neural", target="binary", folds=5, seconds=6, weight=20),
            dict(name="neural_binary_6fold_2s", suite="neural", target="binary", folds=6, seconds=2, weight=18),
            dict(name="neural_three_6fold_2s", suite="neural", target="three", folds=6, seconds=2, weight=15),
            dict(name="neural_three_6fold_6s", suite="neural", target="three", folds=6, seconds=6, weight=15),
            dict(name="neural_binary_6fold_augmented", suite="neural", target="binary", folds=6, seconds=6, weight=14, wave=1),
        ]
    return [
        dict(name="motion_tabular_binary_5fold_2s", suite="tabular", target="binary", folds=5, seconds=2, weight=25),
        dict(name="motion_tabular_binary_5fold_6s", suite="tabular", target="binary", folds=5, seconds=6, weight=25),
        dict(name="motion_tabular_binary_6fold_2s", suite="tabular", target="binary", folds=6, seconds=2, weight=20),
        dict(name="motion_tabular_three_6fold_2s", suite="tabular", target="three", folds=6, seconds=2, weight=15),
        dict(name="motion_tabular_three_6fold_6s", suite="tabular", target="three", folds=6, seconds=6, weight=15),
    ]


def aggregate_reports(base):
    rows = []
    for path in sorted(base.glob("*/summary.json")):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
            row = dict(experiment=path.parent.name, status=s["status"], completed_folds=s["completed_folds"],
                       expected_folds=s["expected_folds"])
            if "fold_metrics" in s:
                row.update(mean_macro_f1=s["fold_metrics"]["macro_f1"]["mean"],
                           fold_sd=s["fold_metrics"]["macro_f1"]["std_sample"])
            rows.append(row)
        except (KeyError, ValueError, OSError):
            continue
    write_json(base / "results_index.json", dict(updated_at=timestamp(), experiments=rows,
               limitation="Do not rank binary and three-class targets or different fold sets together. All comparisons on this cohort remain exploratory."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["laptop", "server"], required=True)
    p.add_argument("--until", default=CUTOFF)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cutoff = datetime.fromisoformat(args.until)
    if cutoff.tzinfo is None:
        raise ValueError("Deadline must be timezone-aware")
    base = BASE / args.role
    base.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "outputs/datasets/research18_yaw_pitch_gaze_v2"
    folds5 = ROOT / "outputs/datasets/research18_yaw_pitch_gaze_v2_folds"
    folds6 = ROOT / "outputs/datasets/research18_yaw_pitch_gaze_v2_6fold_20261006"
    engine = ROOT / "training/overnight_engine.py"
    plan = jobs(args.role)
    declaration = dict(role=args.role, deadline=args.until, jobs=plan, engine=str(engine),
                       python=sys.executable, target_metadata_policy="Labels preserve existing effective condition reviews; condition IDs are targets only")
    declaration_path = base / "queue_plan.json"
    if declaration_path.exists() and json.loads(declaration_path.read_text()) != declaration:
        raise ValueError("Queue declaration differs; use a new experiment directory")
    write_json(declaration_path, declaration)
    if args.dry_run:
        print(json.dumps(declaration, indent=2))
        return
    # Atomic per-device lock. Stale locks can be removed only after verifying PID is gone.
    lock = base / "supervisor.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    prevent_sleep = False
    child = None
    state = dict(role=args.role, supervisor_pid=os.getpid(), until=args.until,
                 status="STARTING", started_at=timestamp(), completed_jobs=[])
    status_path = base / "status.json"
    write_json(status_path, state)
    try:
        if os.name == "nt":
            import ctypes
            # Process-scoped request only; it automatically clears when this thread exits.
            prevent_sleep = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            state["automatic_sleep_prevention"] = prevent_sleep
        subprocess.run([sys.executable, str(engine), "--dataset", str(dataset), "--folds", str(folds6),
                        "--init-sixfold"], check=True, cwd=ROOT, timeout=120)
        for number, job in enumerate(plan):
            remaining = (cutoff-datetime.now().astimezone()).total_seconds()
            if remaining <= 0:
                state["status"] = "DEADLINE_REACHED"
                break
            output = base / job["name"]
            summary_file = output / "summary.json"
            if summary_file.exists() and json.loads(summary_file.read_text())["status"] in ("COMPLETE", "REVIEW"):
                state["completed_jobs"].append(dict(name=job["name"], resumed_complete=True))
                continue
            share = job["weight"] / sum(j["weight"] for j in plan[number:])
            allowed_seconds = max(60., remaining*share - 20.)
            allowed_seconds = min(allowed_seconds, remaining)
            command = [sys.executable, "-u", str(engine), "--dataset", str(dataset),
                "--folds", str(folds5 if job["folds"] == 5 else folds6), "--output", str(output),
                "--suite", job["suite"], "--target", job["target"], "--context-seconds", str(job["seconds"]),
                "--until", args.until, "--hours", str(allowed_seconds/3600), "--threads", "4",
                "--device", "cuda" if args.role == "laptop" else "cpu", "--wave", str(job.get("wave", 0)), "--resume"]
            started = time.monotonic()
            with (base / (job["name"] + ".log")).open("a", encoding="utf-8", buffering=1) as log:
                log.write("\nSUPERVISOR START " + timestamp() + "\n")
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state.update(status="RUNNING", active_job=job["name"], child_pid=child.pid,
                             active_job_started_at=timestamp(), active_job_budget_seconds=allowed_seconds)
                write_json(status_path, state)
                while child.poll() is None:
                    if datetime.now().astimezone() >= cutoff or time.monotonic()-started >= allowed_seconds+10:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=10)
                        state["last_stop_reason"] = "global_deadline" if datetime.now().astimezone() >= cutoff else "stage_time_budget"
                        break
                    state["heartbeat_at"] = timestamp()
                    write_json(status_path, state)
                    time.sleep(5)
                code = child.returncode
                state["completed_jobs"].append(dict(name=job["name"], exit_code=code,
                    elapsed_seconds=time.monotonic()-started, ended_at=timestamp()))
                child = None
                aggregate_reports(base)
                write_json(status_path, state)
        else:
            state["status"] = "QUEUE_COMPLETE"
        state.pop("active_job", None)
        state.pop("child_pid", None)
        state["finished_at"] = timestamp()
        write_json(status_path, state)
        aggregate_reports(base)
    except BaseException as exc:
        state.update(status="ERROR", error=repr(exc), updated_at=timestamp())
        write_json(status_path, state)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=15)
        if prevent_sleep:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
