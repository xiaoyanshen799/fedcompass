from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a FedAvg/FedCompass comparison run by generating client CPU quotas, "
            "updating dataset partition seeds, and printing the exact launch commands."
        )
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed used for quota generation and dataset partition.")
    parser.add_argument(
        "--fit-csv",
        default="examples/output/cpu_theta_runs/concurrent_quota_theta_fit.csv",
        help="Path to concurrent_quota_theta_fit.csv.",
    )
    parser.add_argument(
        "--distribution",
        default="normal",
        choices=["normal", "homogeneous", "homogenous", "exponential"],
        help="Distribution passed to generate_client_quotas_from_fit.py.",
    )
    parser.add_argument(
        "--client-config",
        default="examples/config/client_5.yaml",
        help="Client config used by launch_clients_systemd.sh.",
    )
    parser.add_argument(
        "--server-fedavg-config",
        default="examples/config/server_fedavg.yaml",
        help="FedAvg server config to update.",
    )
    parser.add_argument(
        "--server-fedcompass-config",
        default="examples/config/server_fedcompass.yaml",
        help="FedCompass server config to update.",
    )
    parser.add_argument(
        "--output-root",
        default="examples/output/prepared_runs",
        help="Directory where generated quota files are saved.",
    )
    parser.add_argument(
        "--client-output-fedavg",
        default="normal-5-fedavg-dirichelet",
        help="Client/server output directory name for FedAvg, relative to examples/ unless absolute.",
    )
    parser.add_argument(
        "--client-output-fedcompass",
        default="normal-5-fedcompass-dirichelet",
        help="Client/server output directory name for FedCompass, relative to examples/ unless absolute.",
    )
    parser.add_argument(
        "--python-bin",
        default=".venv/bin/python",
        help="Python interpreter used to run helper scripts.",
    )
    return parser.parse_args()


def resolve_repo_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path)


def resolve_examples_relative(raw: str, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if raw.startswith("examples/"):
        return (repo_root / raw)
    return (repo_root / "examples" / raw)


def update_dataset_seed(config_path: Path, seed: int) -> None:
    cfg = OmegaConf.load(config_path)
    if "server_fed" in config_path.name:
        kwargs = cfg.server_configs.validation_data_configs.dataset_kwargs
    else:
        kwargs = cfg.data_configs.dataset_kwargs
    kwargs.seed = int(seed)
    OmegaConf.save(cfg, config_path)


def generate_quotas(
    repo_root: Path,
    python_bin: Path,
    fit_csv: Path,
    distribution: str,
    seed: int,
    output_root: Path,
) -> tuple[Path, Path, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    output_csv = output_root / f"generated_client_cpu_quotas_{distribution}_seed{seed}.csv"
    output_json = output_root / f"generated_client_cpu_quotas_{distribution}_seed{seed}.json"
    cmd = [
        str(python_bin),
        "examples/generate_client_quotas_from_fit.py",
        "--fit-csv",
        str(fit_csv),
        "--distribution",
        distribution,
        "--seed",
        str(seed),
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    cpu_quota_csv = str(payload["cpu_quota_csv"])
    return output_csv, output_json, cpu_quota_csv


def shell_quote(value: str) -> str:
    escaped = value.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def build_server_command(config_path: Path, output_dir: Path) -> str:
    return (
        "source .venv/bin/activate\n"
        f"cd {config_path.parents[2]}\n"
        f"cd examples\n"
        f"python3 grpc/run_server.py --config {config_path.relative_to(config_path.parents[2] / 'examples')} "
        f"--logging-output-dir {output_dir.relative_to(config_path.parents[2] / 'examples')}"
    )


def build_client_command(
    repo_root: Path,
    client_config: Path,
    output_dir: Path,
    cpu_quota_csv: str,
) -> str:
    examples_dir = repo_root / "examples"
    return (
        "source .venv/bin/activate\n"
        f"cd {repo_root}\n"
        "cd examples\n"
        "./grpc/launch_clients_systemd.sh "
        f"--config {client_config.relative_to(examples_dir)} "
        f"--logging-output-dir {output_dir.relative_to(examples_dir)} "
        f"--data-output-dir {output_dir.relative_to(examples_dir)} "
        "--data-output-filename visualization.pdf "
        f"--cpu-quotas {shell_quote(cpu_quota_csv)}"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args()

    python_bin = resolve_repo_path(args.python_bin, repo_root)
    fit_csv = resolve_repo_path(args.fit_csv, repo_root)
    client_config = resolve_repo_path(args.client_config, repo_root)
    server_fedavg_config = resolve_repo_path(args.server_fedavg_config, repo_root)
    server_fedcompass_config = resolve_repo_path(args.server_fedcompass_config, repo_root)
    output_root = resolve_repo_path(args.output_root, repo_root)
    fedavg_output_dir = resolve_examples_relative(args.client_output_fedavg, repo_root)
    fedcompass_output_dir = resolve_examples_relative(args.client_output_fedcompass, repo_root)

    if not python_bin.exists():
        raise SystemExit(f"Python binary not found: {python_bin}")
    if not fit_csv.exists():
        raise SystemExit(f"Fit CSV not found: {fit_csv}")

    quota_csv_path, quota_json_path, cpu_quota_csv = generate_quotas(
        repo_root=repo_root,
        python_bin=python_bin,
        fit_csv=fit_csv,
        distribution=args.distribution,
        seed=args.seed,
        output_root=output_root,
    )

    update_dataset_seed(client_config, args.seed)
    update_dataset_seed(server_fedavg_config, args.seed)
    update_dataset_seed(server_fedcompass_config, args.seed)

    print(f"Prepared seed: {args.seed}")
    print(f"Generated quota CSV: {quota_csv_path}")
    print(f"Generated quota JSON: {quota_json_path}")
    print(f"CPU quotas: {cpu_quota_csv}")
    print("")
    print("Updated dataset_kwargs.seed in:")
    print(f"  {client_config}")
    print(f"  {server_fedavg_config}")
    print(f"  {server_fedcompass_config}")
    print("")
    print("FedAvg")
    print("Server window:")
    print(build_server_command(server_fedavg_config, fedavg_output_dir))
    print("")
    print("Client window:")
    print(build_client_command(repo_root, client_config, fedavg_output_dir, cpu_quota_csv))
    print("")
    print("FedCompass")
    print("Server window:")
    print(build_server_command(server_fedcompass_config, fedcompass_output_dir))
    print("")
    print("Client window:")
    print(build_client_command(repo_root, client_config, fedcompass_output_dir, cpu_quota_csv))
    print("")
    print("After each server finishes, kill it manually if the process stays alive.")


if __name__ == "__main__":
    main()
