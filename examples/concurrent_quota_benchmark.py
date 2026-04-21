from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def inverse_power_cpu_theta(cpu: np.ndarray, a: float, b: float, p: float) -> np.ndarray:
    return a + b * np.power(cpu, -p)


def inverse_linear_cpu_theta(cpu: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b / cpu


CPU_FIT_SCALE = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run true concurrent FL quota benchmarks with launch_clients_systemd.sh, "
            "then fit per-client CPU -> theta curves from the resulting logs."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["run", "analyze", "both"],
        default="both",
        help="Whether to launch new runs, analyze existing runs, or do both.",
    )
    parser.add_argument(
        "--output-root",
        default="examples/output/concurrent_quota_benchmark",
        help="Root directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="run",
        help="Prefix for generated run subdirectory names.",
    )
    parser.add_argument(
        "--quota-schemes",
        default="",
        help=(
            "Semicolon-separated schemes like "
            "'q1=45,45,45,48,...;q2=35,35,35,40,...'. Required in run/both mode "
            "unless --quota-schemes-file is given."
        ),
    )
    parser.add_argument(
        "--quota-schemes-file",
        default="",
        help=(
            "CSV file with columns run_name,quotas where quotas is a comma-separated "
            "list of per-client CPU percentages."
        ),
    )
    parser.add_argument(
        "--server-config",
        default="config/server_fedavg.yaml",
        help="Server config path relative to examples/ or absolute.",
    )
    parser.add_argument(
        "--client-config",
        default="config/client_1.yaml",
        help="Client config path relative to examples/ or absolute.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=20,
        help="Number of clients to launch in each run.",
    )
    parser.add_argument(
        "--python-bin",
        default="",
        help="Python executable passed to launch_clients_systemd.sh. Defaults to current executable.",
    )
    parser.add_argument(
        "--server-start-delay",
        type=float,
        default=3.0,
        help="Seconds to wait after launching the server before launching clients.",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=7200.0,
        help="Maximum seconds to wait for a benchmark run's server to finish.",
    )
    parser.add_argument(
        "--client-unit-prefix",
        default="quota-bench-client",
        help="Base systemd unit prefix for launched clients.",
    )
    parser.add_argument(
        "--clients-no-sudo",
        action="store_true",
        help="Pass --no-sudo to launch_clients_systemd.sh.",
    )
    parser.add_argument(
        "--time-column",
        default="Time",
        help="Column name containing client training time.",
    )
    parser.add_argument(
        "--round-column",
        default="Round",
        help="Column name containing round index.",
    )
    parser.add_argument(
        "--filter-column",
        default="Pre Val?",
        help="Optional column used to keep only training rows when present.",
    )
    parser.add_argument(
        "--train-flag",
        default="N",
        help="Value in --filter-column indicating a training row.",
    )
    parser.add_argument(
        "--min-round",
        type=float,
        default=1.0,
        help="Minimum round index to include in theta fitting.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional upper bound for time filtering before fitting theta.",
    )
    return parser.parse_args()


def resolve_from_examples(path_str: str, examples_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (examples_dir / path).resolve()


def parse_quotas_string(quotas_text: str) -> list[int]:
    quotas = [int(item.strip().rstrip("%")) for item in quotas_text.split(",") if item.strip()]
    if not quotas:
        raise ValueError("Quota list is empty.")
    return quotas


def parse_scheme_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    schemes: list[dict[str, Any]] = []

    if args.quota_schemes.strip():
        for chunk in args.quota_schemes.split(";"):
            item = chunk.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid scheme: {item}. Expected name=quota1,quota2,...")
            name, quotas_text = item.split("=", maxsplit=1)
            schemes.append(
                {
                    "run_name": name.strip(),
                    "quotas": parse_quotas_string(quotas_text),
                }
            )

    if args.quota_schemes_file.strip():
        path = Path(args.quota_schemes_file).expanduser().resolve()
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "run_name" not in reader.fieldnames or "quotas" not in reader.fieldnames:
                raise ValueError("Quota schemes CSV must have columns: run_name,quotas")
            for row in reader:
                run_name = str(row.get("run_name", "")).strip()
                quotas_text = str(row.get("quotas", "")).strip()
                if not run_name or not quotas_text:
                    continue
                schemes.append(
                    {
                        "run_name": run_name,
                        "quotas": parse_quotas_string(quotas_text),
                    }
                )

    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for scheme in schemes:
        run_name = str(scheme["run_name"])
        if run_name in seen_names:
            raise ValueError(f"Duplicate run_name in schemes: {run_name}")
        seen_names.add(run_name)
        deduped.append(scheme)

    return deduped


def fit_logistic(times: np.ndarray) -> tuple[float, float]:
    x_data = np.sort(times.astype(float))
    y_data = np.arange(1, len(x_data) + 1, dtype=float) / float(len(x_data))

    theta0 = float(np.median(x_data))
    iqr = float(np.percentile(x_data, 75) - np.percentile(x_data, 25))
    k0 = max(iqr / 4.0, 0.1)
    bounds = ([float(x_data.min()), 0.05], [float(x_data.max()), 1000.0])

    popt, _ = curve_fit(
        logistic_cdf,
        x_data,
        y_data,
        p0=[theta0, k0],
        bounds=bounds,
        maxfev=20000,
        loss="soft_l1",
    )
    theta_hat, k_hat = popt
    return float(theta_hat), float(k_hat)


def load_training_times(
    csv_path: Path,
    time_column: str,
    round_column: str,
    filter_column: str,
    train_flag: str,
    max_time: float | None,
    min_round: float,
) -> np.ndarray:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or time_column not in reader.fieldnames:
            raise ValueError(f"missing column: {time_column}")

        use_filter = filter_column in reader.fieldnames
        records: list[tuple[float, float]] = []
        for row in reader:
            if use_filter:
                value = str(row.get(filter_column, "")).strip().upper()
                if value != train_flag.strip().upper():
                    continue

            raw_time = str(row.get(time_column, "")).strip()
            if not raw_time:
                continue

            try:
                time_value = float(raw_time)
            except ValueError:
                continue

            if max_time is not None and time_value > max_time:
                continue

            round_value = float("inf")
            if round_column in row:
                raw_round = str(row.get(round_column, "")).strip()
                if raw_round:
                    try:
                        round_value = float(raw_round)
                    except ValueError:
                        round_value = float("inf")

            if round_value < min_round:
                continue

            records.append((round_value, time_value))

    records.sort(key=lambda item: (item[0], item[1]))
    return np.array([time_value for _, time_value in records], dtype=float)


def discover_run_dirs(output_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if not output_root.exists():
        return run_dirs

    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "client_launch_manifest.csv").exists():
            run_dirs.append(child)
    return run_dirs


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "run"


def create_run_dir(output_root: Path, prefix: str, run_name: str, run_idx: int) -> Path:
    safe_name = sanitize_name(run_name)
    run_dir = output_root / f"{prefix}_{run_idx:02d}_{safe_name}"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def launch_one_run(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    run_name: str,
    quotas: list[int],
    examples_dir: Path,
) -> None:
    if len(quotas) != args.num_clients:
        raise ValueError(
            f"Run {run_name} has {len(quotas)} quotas, but --num-clients={args.num_clients}."
        )

    python_bin = args.python_bin.strip() or sys.executable
    server_config = resolve_from_examples(args.server_config, examples_dir)
    client_config = resolve_from_examples(args.client_config, examples_dir)
    server_stdout = run_dir / "server_stdout.log"
    server_stderr = run_dir / "server_stderr.log"
    manifest_path = run_dir / "client_launch_manifest.csv"
    quota_csv = ",".join(str(quota) for quota in quotas)
    unit_prefix = sanitize_name(f"{args.client_unit_prefix}-{run_name}")

    metadata = {
        "run_name": run_name,
        "num_clients": args.num_clients,
        "quotas": quotas,
        "quota_csv": quota_csv,
        "server_config": str(server_config),
        "client_config": str(client_config),
        "python_bin": python_bin,
        "unit_prefix": unit_prefix,
    }
    write_run_metadata(run_dir, metadata)

    server_cmd = [
        python_bin,
        str(examples_dir / "grpc" / "run_server.py"),
        "--config",
        str(server_config),
        "--logging-output-dir",
        str(run_dir),
        "--logging-output-filename",
        "result",
    ]

    with server_stdout.open("w", encoding="utf-8") as stdout_handle, server_stderr.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=examples_dir,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

    try:
        time.sleep(args.server_start_delay)

        client_cmd = [
            "bash",
            str(examples_dir / "grpc" / "launch_clients_systemd.sh"),
            "--num-clients",
            str(args.num_clients),
            "--config",
            str(client_config),
            "--python",
            str(python_bin),
            "--unit-prefix",
            unit_prefix,
            "--cpu-quotas",
            quota_csv,
            "--logging-output-dir",
            str(run_dir),
            "--logging-output-filename",
            "result",
            "--save-manifest",
            str(manifest_path),
        ]
        if args.clients_no_sudo:
            client_cmd.append("--no-sudo")

        completed = subprocess.run(
            client_cmd,
            cwd=examples_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        (run_dir / "client_launcher_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "client_launcher_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Client launcher failed for run {run_name}. See {run_dir / 'client_launcher_stderr.log'}"
            )

        server_proc.wait(timeout=args.server_timeout)
        (run_dir / "server_exit_code.txt").write_text(str(server_proc.returncode), encoding="utf-8")
        if server_proc.returncode != 0:
            raise RuntimeError(
                f"Server exited with code {server_proc.returncode} for run {run_name}. "
                f"See {server_stderr}"
            )
    except Exception:
        if server_proc.poll() is None:
            server_proc.kill()
            try:
                server_proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        raise


def read_manifest(manifest_path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "client_id" not in reader.fieldnames or "cpu_quota" not in reader.fieldnames:
            raise ValueError(f"Invalid manifest format: {manifest_path}")
        for row in reader:
            client_id = int(str(row["client_id"]).strip())
            rows[client_id] = {str(key): str(value) for key, value in row.items()}
    return rows


def parse_result_csv_name(path: Path) -> tuple[int, str] | None:
    match = re.fullmatch(r"result_Client(\d+)(?:_(.+))?", path.stem)
    if match is None:
        return None
    return int(match.group(1)), match.group(2) or ""


def analyze_one_run(run_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = run_dir / "client_launch_manifest.csv"
    if not manifest_path.exists():
        return []

    manifest = read_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(run_dir.glob("result_Client*.csv")):
        parsed = parse_result_csv_name(csv_path)
        if parsed is None:
            continue
        client_id, suffix = parsed
        if suffix:
            continue
        if client_id not in manifest:
            continue

        times = load_training_times(
            csv_path=csv_path,
            time_column=args.time_column,
            round_column=args.round_column,
            filter_column=args.filter_column,
            train_flag=args.train_flag,
            max_time=args.max_time,
            min_round=args.min_round,
        )
        if times.size == 0:
            continue

        theta_hat, k_hat = fit_logistic(times)
        manifest_row = manifest[client_id]
        cpu_quota_text = manifest_row["cpu_quota"].strip()
        cpu_quota = float(cpu_quota_text.rstrip("%"))
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "client_id": client_id,
                "client": f"Client{client_id}",
                "cpu_quota": cpu_quota,
                "cpu_quota_text": cpu_quota_text,
                "num_samples": int(times.size),
                "theta": theta_hat,
                "k": k_hat,
                "time_mean": float(times.mean()),
                "time_min": float(times.min()),
                "time_max": float(times.max()),
                "csv_path": str(csv_path),
            }
        )

    rows.sort(key=lambda row: int(row["client_id"]))
    return rows


def fit_one_client_curve(client_rows: list[dict[str, Any]]) -> dict[str, Any]:
    client_id = int(client_rows[0]["client_id"])
    client_name = str(client_rows[0]["client"])
    x = np.array([float(row["cpu_quota"]) for row in client_rows], dtype=float)
    y = np.array([float(row["theta"]) for row in client_rows], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_fit = x / CPU_FIT_SCALE

    if len(np.unique(x)) < 2:
        return {
            "client_id": client_id,
            "client": client_name,
            "num_points": len(x),
            "fit_model": "insufficient_data",
            "a": np.nan,
            "b": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "cpu_min": float(x.min()),
            "cpu_max": float(x.max()),
            "cpu_fit_scale": CPU_FIT_SCALE,
            "theta_min": float(y.min()),
            "theta_max": float(y.max()),
        }

    try:
        p0 = [max(float(y.min()) * 0.8, 0.0), max((float(y.max()) - float(y.min())) * 0.2, 0.1), 1.0]
        bounds = ([0.0, 0.0, 0.1], [max(float(y.max()) * 2.0, 1.0), 1e6, 5.0])
        popt, _ = curve_fit(
            inverse_power_cpu_theta,
            x_fit,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        predictions = inverse_power_cpu_theta(x_fit, *popt)
        fit_model = "inverse_power"
        a_hat, b_hat, p_hat = [float(value) for value in popt]
    except Exception:  # noqa: BLE001
        popt, _ = curve_fit(
            inverse_linear_cpu_theta,
            x_fit,
            y,
            p0=[max(float(y.min()) * 0.8, 0.0), max((float(y.max()) - float(y.min())) * 0.2, 0.1)],
            bounds=([0.0, 0.0], [max(float(y.max()) * 2.0, 1.0), 1e6]),
            maxfev=20000,
        )
        predictions = inverse_linear_cpu_theta(x_fit, *popt)
        fit_model = "inverse_linear"
        a_hat, b_hat = [float(value) for value in popt]
        p_hat = 1.0

    residual = y - predictions
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y - y.mean())))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    return {
        "client_id": client_id,
        "client": client_name,
        "num_points": len(x),
        "fit_model": fit_model,
        "a": a_hat,
        "b": b_hat,
        "p": p_hat,
        "r2": r2,
        "cpu_min": float(x.min()),
        "cpu_max": float(x.max()),
        "cpu_fit_scale": CPU_FIT_SCALE,
        "theta_min": float(y.min()),
        "theta_max": float(y.max()),
    }


def build_curve_prediction_rows(
    client_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fit_map = {int(row["client_id"]): row for row in fit_rows}
    prediction_rows: list[dict[str, Any]] = []
    for row in client_rows:
        client_id = int(row["client_id"])
        fit_row = fit_map.get(client_id)
        if fit_row is None or str(fit_row["fit_model"]) == "insufficient_data":
            predicted_theta = np.nan
        elif str(fit_row["fit_model"]) == "inverse_power":
            predicted_theta = float(
                inverse_power_cpu_theta(
                    np.array([float(row["cpu_quota"]) / float(fit_row.get("cpu_fit_scale", 1.0))], dtype=float),
                    float(fit_row["a"]),
                    float(fit_row["b"]),
                    float(fit_row["p"]),
                )[0]
            )
        else:
            predicted_theta = float(
                inverse_linear_cpu_theta(
                    np.array([float(row["cpu_quota"]) / float(fit_row.get("cpu_fit_scale", 1.0))], dtype=float),
                    float(fit_row["a"]),
                    float(fit_row["b"]),
                )[0]
            )
        enriched = dict(row)
        enriched["predicted_theta"] = predicted_theta
        enriched["prediction_error"] = (
            float(row["theta"]) - predicted_theta if not math.isnan(predicted_theta) else np.nan
        )
        prediction_rows.append(enriched)
    return prediction_rows


def plot_client_curves(
    client_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    plots_dir: Path,
) -> list[str]:
    if plt is None:
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in client_rows:
        grouped.setdefault(int(row["client_id"]), []).append(row)
    fit_map = {int(row["client_id"]): row for row in fit_rows}

    for client_id in sorted(grouped):
        fit_row = fit_map.get(client_id)
        if fit_row is None:
            continue
        rows = sorted(grouped[client_id], key=lambda row: float(row["cpu_quota"]))
        x = np.array([float(row["cpu_quota"]) for row in rows], dtype=float)
        y = np.array([float(row["theta"]) for row in rows], dtype=float)

        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, label="Observed theta", color="#1f77b4")

        if str(fit_row["fit_model"]) != "insufficient_data":
            x_grid = np.linspace(float(x.min()), float(x.max()), 200)
            x_grid_fit = x_grid / float(fit_row.get("cpu_fit_scale", 1.0))
            if str(fit_row["fit_model"]) == "inverse_power":
                y_grid = inverse_power_cpu_theta(
                    x_grid_fit,
                    float(fit_row["a"]),
                    float(fit_row["b"]),
                    float(fit_row["p"]),
                )
            else:
                y_grid = inverse_linear_cpu_theta(
                    x_grid_fit,
                    float(fit_row["a"]),
                    float(fit_row["b"]),
                )
            plt.plot(x_grid, y_grid, color="#d62728", label=str(fit_row["fit_model"]))

        plt.xlabel("CPU Quota (%)")
        plt.ylabel("Theta (s)")
        plt.title(f"Client {client_id} CPU -> Theta")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = plots_dir / f"client_{client_id:02d}_cpu_theta.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        saved_paths.append(str(out_path))
    return saved_paths


def write_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmarks(args: argparse.Namespace) -> None:
    examples_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    schemes = parse_scheme_specs(args)
    if not schemes:
        raise SystemExit("No quota schemes provided. Use --quota-schemes or --quota-schemes-file.")

    for run_idx, scheme in enumerate(schemes, start=1):
        run_name = str(scheme["run_name"])
        quotas = list(scheme["quotas"])
        run_dir = create_run_dir(output_root, args.run_name_prefix, run_name, run_idx)
        print(f"[run {run_idx}/{len(schemes)}] {run_name} -> {run_dir}")
        launch_one_run(
            args=args,
            run_dir=run_dir,
            run_name=run_name,
            quotas=quotas,
            examples_dir=examples_dir,
        )
        print(f"[done] {run_name}")


def analyze_benchmarks(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    run_dirs = discover_run_dirs(output_root)
    if not run_dirs:
        raise SystemExit(f"No run directories with client_launch_manifest.csv found under {output_root}")

    all_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_rows = analyze_one_run(run_dir, args)
        all_rows.extend(run_rows)

    if not all_rows:
        raise SystemExit("No client result rows were extracted from the discovered runs.")

    all_rows.sort(key=lambda row: (int(row["client_id"]), str(row["run_name"])))
    raw_csv = output_root / "concurrent_quota_theta_raw.csv"
    write_csv(
        all_rows,
        raw_csv,
        [
            "run_dir",
            "run_name",
            "client_id",
            "client",
            "cpu_quota",
            "cpu_quota_text",
            "num_samples",
            "theta",
            "k",
            "time_mean",
            "time_min",
            "time_max",
            "csv_path",
        ],
    )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault(int(row["client_id"]), []).append(row)

    fit_rows = [fit_one_client_curve(grouped[client_id]) for client_id in sorted(grouped)]
    fit_csv = output_root / "concurrent_quota_theta_fit.csv"
    write_csv(
        fit_rows,
        fit_csv,
        [
            "client_id",
            "client",
            "num_points",
            "fit_model",
            "a",
            "b",
            "p",
            "r2",
            "cpu_min",
            "cpu_max",
            "cpu_fit_scale",
            "theta_min",
            "theta_max",
        ],
    )

    prediction_rows = build_curve_prediction_rows(all_rows, fit_rows)
    prediction_csv = output_root / "concurrent_quota_theta_predictions.csv"
    write_csv(
        prediction_rows,
        prediction_csv,
        [
            "run_dir",
            "run_name",
            "client_id",
            "client",
            "cpu_quota",
            "cpu_quota_text",
            "num_samples",
            "theta",
            "predicted_theta",
            "prediction_error",
            "k",
            "time_mean",
            "time_min",
            "time_max",
            "csv_path",
        ],
    )

    plot_paths = plot_client_curves(all_rows, fit_rows, output_root / "plots")

    print(f"Analyzed {len(run_dirs)} runs.")
    print(f"Saved raw theta rows to: {raw_csv}")
    print(f"Saved fitted CPU->theta parameters to: {fit_csv}")
    print(f"Saved prediction diagnostics to: {prediction_csv}")
    if plot_paths:
        print(f"Saved {len(plot_paths)} client curve plots under: {output_root / 'plots'}")
    else:
        print("Skipped plotting because matplotlib is unavailable.")


def main() -> None:
    args = parse_args()
    if args.mode in {"run", "both"}:
        run_benchmarks(args)
    if args.mode in {"analyze", "both"}:
        analyze_benchmarks(args)


if __name__ == "__main__":
    main()
