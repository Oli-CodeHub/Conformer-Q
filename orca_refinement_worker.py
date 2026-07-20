#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ENERGY_PATTERN = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
GIBBS_PATTERN = re.compile(r"Final Gibbs free energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh")
FREQUENCY_PATTERN = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", re.MULTILINE)


def orca_executable_path(value: str) -> Path:
    return Path(value).expanduser()


def write_task(path: Path, task: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    orca_bin = orca_executable_path(sys.argv[2])
    task_path = run_dir / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PATH"] = f"{orca_bin.parent}:{env.get('PATH', '')}"
    selected = task["selected_conformers"]
    completed = []
    try:
        for index, conformer in enumerate(selected, start=1):
            conformer_started_at = time.time()
            print(f"Starting ORCA refinement {index}/{len(selected)} for source conformer #{conformer['source_rank']}", flush=True)
            task["current_index"] = index
            task["current_conformer_started_at"] = conformer_started_at
            task["progress"] = f"正在精修代表构象 {index} / {len(selected)}（原始构象 #{conformer['source_rank']}）..."
            write_task(task_path, task)
            prefix = f"conf_{index:03d}"
            inp = run_dir / f"{prefix}.inp"
            out = run_dir / f"{prefix}.out"
            with out.open("w", encoding="utf-8") as output:
                completed_process = subprocess.run(
                    [str(orca_bin), inp.name],
                    cwd=run_dir,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            text = out.read_text(encoding="utf-8", errors="ignore")
            match = ENERGY_PATTERN.findall(text)
            gibbs = GIBBS_PATTERN.findall(text)
            if completed_process.returncode != 0 or "ORCA TERMINATED NORMALLY" not in text or not match or not gibbs:
                raise RuntimeError(f"构象 #{conformer['source_rank']} 的 ORCA 优化或频率计算未正常完成。")
            significant_imaginary = [float(value) for value in FREQUENCY_PATTERN.findall(text) if float(value) < -20.0]
            conformer_completed_at = time.time()
            completed.append(
                {
                    "source_rank": conformer["source_rank"],
                    "family_id": conformer["family_id"],
                    "energy_hartree": float(match[-1]),
                    "gibbs_free_energy_hartree": float(gibbs[-1]),
                    "significant_imaginary_frequencies_cm_1": significant_imaginary,
                    "xyz_file": f"{prefix}.xyz",
                    "output_file": out.name,
                    "started_at": conformer_started_at,
                    "completed_at": conformer_completed_at,
                    "duration_seconds": max(0, int(conformer_completed_at - conformer_started_at)),
                }
            )
            print(
                f"Completed source conformer #{conformer['source_rank']}: "
                f"E={float(match[-1]):.10f} Eh G={float(gibbs[-1]):.10f} Eh",
                flush=True,
            )
            task["completed_conformers"] = completed
            write_task(task_path, task)
        task["status"] = "completed"
        task.pop("current_conformer_started_at", None)
        task["progress"] = "ORCA 优化、频率计算与自由能排序已完成，可查看最终集合。"
        print("ORCA refinement task completed normally.", flush=True)
    except Exception as exc:
        task["status"] = "failed"
        task["progress"] = "ORCA 精修未正常完成，请查看输出日志。"
        task["error"] = str(exc)
        print(f"ORCA refinement failed: {exc}", flush=True)
    task["completed_at"] = time.time()
    task["duration_seconds"] = max(0, int(task["completed_at"] - task["created_at"]))
    write_task(task_path, task)
    return 0 if task["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
