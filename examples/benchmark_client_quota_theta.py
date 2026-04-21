from __future__ import annotations

import argparse
import csv
import json
import os
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import curve_fit


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def fit_logistic(times: list[float]) -> tuple[float, float, str]:
    x_data = np.sort(np.asarray(times, dtype=float))
    if x_data.size == 0:
        raise ValueError("No timing samples to fit.")
    if x_data.size < 4:
        return float(np.median(x_data)), 0.0, "median_fallback"

    y_data = np.arange(1, len(x_data) + 1, dtype=float) / float(len(x_data))
    theta0 = float(np.median(x_data))
    iqr = float(np.percentile(x_data, 75) - np.percentile(x_data, 25))
    k0 = max(iqr / 4.0, 0.05)
    bounds = ([float(x_data.min()), 0.01], [float(x_data.max()), 1000.0])

    try:
        popt, _ = curve_fit(
            logistic_cdf,
            x_data,
            y_data,
            p0=[theta0, k0],
            bounds=bounds,
            maxfev=20000,
            loss="soft_l1",
        )
    except Exception:
        return float(np.median(x_data)), 0.0, "median_fallback"

    theta_hat, k_hat = popt
    return float(theta_hat), float(k_hat), "logistic"


def parse_client_ids(spec: str) -> list[int]:
    client_ids: list[int] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", maxsplit=1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            client_ids.extend(list(range(start, end + step, step)))
        else:
            client_ids.append(int(item))
    deduped = sorted(set(client_ids))
    if not deduped:
        raise ValueError("No client ids specified.")
    return deduped


def parse_quotas(spec: str) -> list[int]:
    quotas = [int(item.strip().rstrip("%")) for item in spec.split(",") if item.strip()]
    if not quotas:
        raise ValueError("No CPU quotas specified.")
    return quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the same client under multiple CPU quotas and fit timing theta "
            "for each (client, quota) pair."
        )
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
        "--client-ids",
        default="1-20",
        help="Client ids to benchmark, e.g. 1-20 or 1,3,5.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=20,
        help="Total client count used by the dataset partitioner.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Local training steps per timed run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="How many timed runs to collect per (client, quota).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="How many untimed warmup runs to discard before timing.",
    )
    parser.add_argument(
        "--quotas",
        default="80,70,60,50,40,30,20",
        help="Comma-separated CPUQuota percentages.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/output/quota_theta_benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--output-csv",
        default="quota_theta_results.csv",
        help="Filename for structured results inside --output-dir.",
    )
    parser.add_argument(
        "--metadata-json",
        default="quota_theta_metadata.json",
        help="Filename for run metadata inside --output-dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for repeatable client setup.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal mode: run a single (client, quota) benchmark and print JSON.",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="Internal mode metadata: 1-based client id.",
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=None,
        help="Internal mode metadata: CPU quota percentage.",
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
    client_id: int,
    num_clients: int,
    steps: int,
    seed: int,
) -> object:
    client_cfg = OmegaConf.load(client_config_path)
    server_cfg = OmegaConf.load(server_config_path)
    merged_cfg = OmegaConf.merge(client_cfg, server_cfg.client_configs)

    merged_cfg.train_configs.seed = seed
    merged_cfg.train_configs.num_local_steps = steps
    merged_cfg.train_configs.do_validation = False
    merged_cfg.train_configs.do_pre_validation = False
    merged_cfg.train_configs.logging_output_dirname = ""
    merged_cfg.train_configs.logging_output_filename = ""
    merged_cfg.train_configs.logging_id = f"Client{client_id}"

    merged_cfg.data_configs.dataset_kwargs.num_clients = num_clients
    merged_cfg.data_configs.dataset_kwargs.client_id = client_id - 1
    merged_cfg.data_configs.dataset_kwargs.visualization = False

    return merged_cfg


def run_worker(args: argparse.Namespace) -> None:
    if args.client_id is None or args.quota is None:
        raise SystemExit("--worker requires --client-id and --quota")

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

    raw_runs: list[float] = []
    total_runs = args.warmup + args.repeats
    for run_idx in range(total_runs):
        client_cfg = build_client_config(
            client_config_path=client_config_path,
            server_config_path=server_config_path,
            client_id=args.client_id,
            num_clients=args.num_clients,
            steps=args.steps,
            seed=args.seed,
        )
        client_agent = APPFLClientAgent(client_agent_config=client_cfg)
        if hasattr(client_agent, "logger") and hasattr(client_agent.logger, "logger"):
            client_agent.logger.logger.setLevel(logging.ERROR)
            for handler in client_agent.logger.logger.handlers:
                handler.setLevel(logging.ERROR)
        start = time.perf_counter()
        client_agent.train()
        elapsed = time.perf_counter() - start
        if run_idx >= args.warmup:
            raw_runs.append(elapsed)

    theta, k, fit_method = fit_logistic(raw_runs)
    result = {
        "client_id": args.client_id,
        "quota_percent": args.quota,
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "theta": theta,
        "k": k,
        "fit_method": fit_method,
        "avg_duration_s": float(np.mean(raw_runs)),
        "std_duration_s": float(np.std(raw_runs)),
        "avg_s_per_step": float(np.mean(raw_runs) / args.steps),
        "runs_s": raw_runs,
    }
    print(json.dumps(result))


def run_parent(args: argparse.Namespace) -> None:
    examples_dir = Path(__file__).resolve().parent
    script_path = Path(__file__).resolve()
    client_ids = parse_client_ids(args.client_ids)
    quotas = parse_quotas(args.quotas)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / args.output_csv
    metadata_json = output_dir / args.metadata_json

    metadata = {
        "client_config": str(resolve_config(args.client_config, examples_dir)),
        "server_config": str(resolve_config(args.server_config, examples_dir)),
        "client_ids": client_ids,
        "num_clients": args.num_clients,
        "steps": args.steps,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "quotas": quotas,
        "seed": args.seed,
        "python": sys.executable,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    total_jobs = len(client_ids) * len(quotas)
    job_idx = 0
    for client_id in client_ids:
        for quota in quotas:
            job_idx += 1
            print(
                f"[{job_idx}/{total_jobs}] client={client_id} quota={quota}% "
                f"steps={args.steps} repeats={args.repeats}",
                flush=True,
            )
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
                "--client-config",
                str(resolve_config(args.client_config, examples_dir)),
                "--server-config",
                str(resolve_config(args.server_config, examples_dir)),
                "--client-id",
                str(client_id),
                "--num-clients",
                str(args.num_clients),
                "--quota",
                str(quota),
                "--steps",
                str(args.steps),
                "--repeats",
                str(args.repeats),
                "--warmup",
                str(args.warmup),
                "--seed",
                str(args.seed),
            ]
            completed = subprocess.run(
                cmd,
                cwd=examples_dir,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                print(f"[failed] client={client_id} quota={quota}%")
                if completed.stdout.strip():
                    print(completed.stdout.strip())
                if completed.stderr.strip():
                    print(completed.stderr.strip())
                continue

            json_line = None
            for line in completed.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    json_line = stripped
            if json_line is None:
                print(f"[failed] client={client_id} quota={quota}%: missing JSON result")
                if completed.stdout.strip():
                    print(completed.stdout.strip())
                continue

            result = json.loads(json_line)
            rows.append(result)
            print(
                f"  theta={result['theta']:.6f} k={result['k']:.6f} "
                f"avg_s_per_step={result['avg_s_per_step']:.6f}"
            )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client_id",
                "quota_percent",
                "steps",
                "warmup",
                "repeats",
                "seed",
                "theta",
                "k",
                "fit_method",
                "avg_duration_s",
                "std_duration_s",
                "avg_s_per_step",
                "runs_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            dumped = dict(row)
            dumped["runs_s"] = json.dumps(row["runs_s"])
            writer.writerow(dumped)

    print(f"\nSaved results to: {output_csv}")
    print(f"Saved metadata to: {metadata_json}")


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
