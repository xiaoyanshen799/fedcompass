from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark client training seconds-per-step under different CPU quotas."
    )
    parser.add_argument(
        "--client-config",
        default="config/client_1.yaml",
        help="Client config path, relative to examples/ or absolute.",
    )
    parser.add_argument(
        "--server-config",
        default="config/server_fedavg.yaml",
        help="Server config path, relative to examples/ or absolute.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Local training steps per timing run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many timed runs to average.",
    )
    parser.add_argument(
        "--quotas",
        default="100,70,50,30",
        help="Comma-separated CPUQuota percentages for batch mode.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal mode: run a single benchmark and print JSON.",
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=None,
        help="Internal mode metadata: the CPU quota used for this worker.",
    )
    return parser.parse_args()


def resolve_config(path_str: str, examples_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (examples_dir / path).resolve()


def prepare_single_thread_env() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def build_client_config(
    client_config_path: Path,
    server_config_path: Path,
) -> object:
    client_cfg = OmegaConf.load(client_config_path)
    server_cfg = OmegaConf.load(server_config_path)
    merged_cfg = OmegaConf.merge(client_cfg, server_cfg.client_configs)

    # Benchmark pure local training only.
    merged_cfg.train_configs.do_validation = False
    merged_cfg.train_configs.do_pre_validation = False
    merged_cfg.train_configs.logging_output_dirname = ""
    merged_cfg.train_configs.logging_output_filename = ""
    merged_cfg.data_configs.dataset_kwargs.visualization = False

    return merged_cfg


def run_worker(args: argparse.Namespace) -> None:
    examples_dir = Path(__file__).resolve().parent
    os.chdir(examples_dir)
    prepare_single_thread_env()

    import torch
    from appfl.agent import APPFLClientAgent

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    client_config_path = resolve_config(args.client_config, examples_dir)
    server_config_path = resolve_config(args.server_config, examples_dir)
    client_cfg = build_client_config(client_config_path, server_config_path)
    client_cfg.train_configs.num_local_steps = args.steps

    client_agent = APPFLClientAgent(client_agent_config=client_cfg)

    # Silence per-run training logs in benchmark mode.
    if hasattr(client_agent, "logger") and hasattr(client_agent.logger, "logger"):
        client_agent.logger.logger.setLevel(logging.ERROR)
        for handler in client_agent.logger.logger.handlers:
            handler.setLevel(logging.ERROR)

    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        client_agent.train()
        durations.append(time.perf_counter() - start)

    avg_duration = sum(durations) / len(durations)
    result = {
        "quota_percent": args.quota,
        "steps": args.steps,
        "repeats": args.repeats,
        "avg_duration_s": avg_duration,
        "avg_s_per_step": avg_duration / args.steps,
        "runs_s": durations,
    }
    print(json.dumps(result))


def run_parent(args: argparse.Namespace) -> None:
    examples_dir = Path(__file__).resolve().parent
    script_path = Path(__file__).resolve()
    quotas = [int(item.strip()) for item in args.quotas.split(",") if item.strip()]

    for quota in quotas:
        cmd = [
            "systemd-run",
            "--user",
            "--wait",
            "--pipe",
            "-p",
            f"CPUQuota={quota}%",
            sys.executable,
            str(script_path),
            "--worker",
            "--quota",
            str(quota),
            "--client-config",
            str(resolve_config(args.client_config, examples_dir)),
            "--server-config",
            str(resolve_config(args.server_config, examples_dir)),
            "--steps",
            str(args.steps),
            "--repeats",
            str(args.repeats),
        ]
        print(f"Running quota {quota}% ...", flush=True)
        completed = subprocess.run(
            cmd,
            cwd=examples_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            print(f"[failed] quota={quota}%")
            if completed.stdout.strip():
                print(completed.stdout.strip())
            if completed.stderr.strip():
                print(completed.stderr.strip())
            continue

        json_line = None
        for line in completed.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_line = line
        if json_line is None:
            print(f"[failed] quota={quota}%: missing JSON result")
            if completed.stdout.strip():
                print(completed.stdout.strip())
            continue

        result = json.loads(json_line)
        print(
            f"quota={quota}% avg_s_per_step={result['avg_s_per_step']:.6f} "
            f"avg_duration_s={result['avg_duration_s']:.6f}"
        )


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
