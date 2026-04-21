from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-client logistic theta for each batch, then estimate the minimum "
            "CPU quota needed to slow each client down to that batch's slowest theta."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["examples/normal-dirichlet20clientsuneven/result_Client1*.csv"],
        help=(
            "CSV files or glob patterns used to identify one or more batches. "
            "Passing result_Client1_3.csv loads the whole batch *_3.csv from the same directory."
        ),
    )
    parser.add_argument(
        "--quota-script",
        default="examples/grpc/launch_clients_systemd.sh",
        help="Shell script containing CLIENT_CPU_QUOTAS.",
    )
    parser.add_argument(
        "--output-csv",
        default="examples/output/client_theta_cpu_alignment.csv",
        help="Path to save per-batch theta and recommended CPU quotas.",
    )
    parser.add_argument(
        "--output-summary",
        default="examples/output/client_theta_cpu_alignment_summary.csv",
        help="Path to save per-client averages across batches.",
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
        help="Optional column used to keep only training rows.",
    )
    parser.add_argument(
        "--train-flag",
        default="N",
        help="Value in --filter-column indicating a training row.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional upper bound for time filtering.",
    )
    parser.add_argument(
        "--min-round",
        type=float,
        default=1.0,
        help="Minimum round index to include. Default drops round 0 from fitting.",
    )
    parser.add_argument(
        "--min-cpu",
        type=float,
        default=1.0,
        help="Lower bound for recommended CPU quota percentage.",
    )
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        expanded_pattern = str(Path(pattern).expanduser())
        matches = sorted(Path(match).resolve() for match in glob.glob(expanded_pattern, recursive=True))
        if matches:
            for match in matches:
                if match.is_file() and match not in seen:
                    files.append(match)
                    seen.add(match)
            continue

        candidate = Path(expanded_pattern).resolve()
        if candidate.is_file() and candidate not in seen:
            files.append(candidate)
            seen.add(candidate)

    return files


def parse_result_csv_name(path: Path) -> tuple[int, str] | None:
    match = re.fullmatch(r"result_Client(\d+)(?:_(.+))?", path.stem)
    if match is None:
        return None
    return int(match.group(1)), match.group(2) or ""


def batch_sort_key(batch_key: str) -> tuple[int, int | str]:
    if batch_key == "":
        return (0, 0)
    if batch_key.isdigit():
        return (1, int(batch_key))
    return (2, batch_key)


def batch_label_from_key(batch_key: str) -> str:
    return "base" if batch_key == "" else batch_key


def collect_batch_files(csv_files: list[Path]) -> dict[str, list[Path]]:
    batches: dict[str, dict[int, Path]] = {}

    for csv_path in csv_files:
        parsed = parse_result_csv_name(csv_path)
        if parsed is None:
            continue
        _, batch_key = parsed
        pattern = f"result_Client*{f'_{batch_key}' if batch_key else ''}.csv"
        for match in sorted(csv_path.parent.glob(pattern)):
            parsed_match = parse_result_csv_name(match)
            if parsed_match is None:
                continue
            client_id, match_batch_key = parsed_match
            if match_batch_key != batch_key:
                continue
            batches.setdefault(batch_key, {})[client_id] = match.resolve()

    ordered_batches: dict[str, list[Path]] = {}
    for batch_key in sorted(batches, key=batch_sort_key):
        ordered_batches[batch_key] = [
            path for _, path in sorted(batches[batch_key].items(), key=lambda item: item[0])
        ]
    return ordered_batches


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

        records: list[tuple[float, float]] = []
        use_filter = filter_column in reader.fieldnames

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


def parse_quota_script(quota_script: Path) -> dict[int, float]:
    text = quota_script.read_text(encoding="utf-8")
    matches = re.findall(r"^[ \t]*CLIENT_CPU_QUOTAS=\((.*?)\)[ \t]*$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"CLIENT_CPU_QUOTAS not found in {quota_script}")

    body = matches[-1]
    normalized_body = body.replace(",", " ")
    tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|([^\s]+)', normalized_body)
    values: list[str] = []
    for quoted_double, quoted_single, bare in tokens:
        token = quoted_double or quoted_single or bare
        token = token.strip().rstrip(",")
        if token:
            values.append(token)

    quotas: dict[int, float] = {}
    for idx, value in enumerate(values, start=1):
        quotas[idx] = parse_percent(value)
    return quotas


def parse_percent(value: str) -> float:
    stripped = value.strip()
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    return float(stripped)


def format_percent(value: float) -> str:
    rounded = round(value, 4)
    if math.isclose(rounded, round(rounded)):
        return f"{int(round(rounded))}%"
    return f"{rounded:.4f}".rstrip("0").rstrip(".") + "%"


def estimate_target_cpu(current_cpu: float, current_theta: float, target_theta: float, min_cpu: float) -> float:
    if target_theta <= 0.0 or current_theta <= 0.0:
        return current_cpu
    recommended = current_cpu * current_theta / target_theta
    recommended = min(recommended, current_cpu)
    return max(recommended, min_cpu)


def summarize_batch(
    batch_key: str,
    batch_files: list[Path],
    args: argparse.Namespace,
    quotas: dict[int, float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    for csv_path in batch_files:
        parsed = parse_result_csv_name(csv_path)
        if parsed is None:
            continue
        client_id, _ = parsed
        current_cpu = quotas.get(client_id)
        if current_cpu is None:
            raise ValueError(
                f"Client{client_id} has no matching CPU quota in {args.quota_script}. "
                "Check the quota array length."
            )

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
            raise ValueError(f"{csv_path} has no usable training rows after filtering.")

        theta_hat, k_hat = fit_logistic(times)
        rows.append(
            {
                "batch": batch_label_from_key(batch_key),
                "client_id": client_id,
                "client": f"Client{client_id}",
                "num_samples": int(times.size),
                "current_cpu_quota": current_cpu,
                "theta": theta_hat,
                "k": k_hat,
                "time_mean": float(times.mean()),
                "time_min": float(times.min()),
                "time_max": float(times.max()),
            }
        )

    if not rows:
        return rows

    slowest_theta = max(float(row["theta"]) for row in rows)
    for row in rows:
        theta = float(row["theta"])
        current_cpu = float(row["current_cpu_quota"])
        recommended_cpu = estimate_target_cpu(
            current_cpu=current_cpu,
            current_theta=theta,
            target_theta=slowest_theta,
            min_cpu=args.min_cpu,
        )
        row["target_theta"] = slowest_theta
        row["recommended_cpu_quota"] = recommended_cpu
        row["cpu_reduction"] = current_cpu - recommended_cpu
        row["cpu_reduction_ratio"] = (
            (current_cpu - recommended_cpu) / current_cpu if current_cpu > 0.0 else 0.0
        )

    rows.sort(key=lambda row: int(row["client_id"]))
    return rows


def write_rows(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch",
                "client_id",
                "client",
                "num_samples",
                "current_cpu_quota",
                "theta",
                "k",
                "time_mean",
                "time_min",
                "time_max",
                "target_theta",
                "recommended_cpu_quota",
                "cpu_reduction",
                "cpu_reduction_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[int, list[dict[str, float | int | str]]] = {}
    for row in rows:
        client_id = int(row["client_id"])
        grouped.setdefault(client_id, []).append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for client_id in sorted(grouped):
        client_rows = grouped[client_id]
        current_cpu = float(client_rows[0]["current_cpu_quota"])
        theta_values = np.array([float(row["theta"]) for row in client_rows], dtype=float)
        target_values = np.array([float(row["target_theta"]) for row in client_rows], dtype=float)
        recommended_values = np.array(
            [float(row["recommended_cpu_quota"]) for row in client_rows],
            dtype=float,
        )
        reduction_values = np.array([float(row["cpu_reduction"]) for row in client_rows], dtype=float)

        summary_rows.append(
            {
                "client_id": client_id,
                "client": f"Client{client_id}",
                "num_batches": len(client_rows),
                "current_cpu_quota": current_cpu,
                "mean_theta": float(theta_values.mean()),
                "mean_target_theta": float(target_values.mean()),
                "mean_recommended_cpu_quota": float(recommended_values.mean()),
                "min_recommended_cpu_quota": float(recommended_values.min()),
                "max_recommended_cpu_quota": float(recommended_values.max()),
                "mean_cpu_reduction": float(reduction_values.mean()),
            }
        )
    return summary_rows


def write_summary(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client_id",
                "client",
                "num_batches",
                "current_cpu_quota",
                "mean_theta",
                "mean_target_theta",
                "mean_recommended_cpu_quota",
                "min_recommended_cpu_quota",
                "max_recommended_cpu_quota",
                "mean_cpu_reduction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_batch_console_summary(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return

    batch_name = str(rows[0]["batch"])
    target_theta = float(rows[0]["target_theta"])
    suggested = ", ".join(format_percent(float(row["recommended_cpu_quota"])) for row in rows)
    print(f"[batch={batch_name}] slowest theta = {target_theta:.4f}")
    print(f"[batch={batch_name}] suggested CPU quotas = [{suggested}]")


def print_average_console_summary(summary_rows: list[dict[str, float | int | str]]) -> None:
    print("\nPer-client average suggested quota across batches:")
    for row in summary_rows:
        print(
            f"  Client{int(row['client_id']):02d}: "
            f"current={format_percent(float(row['current_cpu_quota']))}, "
            f"mean_recommended={format_percent(float(row['mean_recommended_cpu_quota']))}, "
            f"mean_theta={float(row['mean_theta']):.4f}, "
            f"mean_target_theta={float(row['mean_target_theta']):.4f}"
        )


def main() -> None:
    args = parse_args()
    csv_files = expand_inputs(args.inputs)
    if not csv_files:
        raise SystemExit("No CSV files found.")

    batch_files_map = collect_batch_files(csv_files)
    if not batch_files_map:
        raise SystemExit(
            "No batch-style client CSVs found. Expected names like result_Client1.csv or result_Client1_5.csv."
        )

    quota_script = Path(args.quota_script).resolve()
    quotas = parse_quota_script(quota_script)

    all_rows: list[dict[str, float | int | str]] = []
    for batch_key, batch_files in batch_files_map.items():
        batch_rows = summarize_batch(
            batch_key=batch_key,
            batch_files=batch_files,
            args=args,
            quotas=quotas,
        )
        print_batch_console_summary(batch_rows)
        all_rows.extend(batch_rows)

    if not all_rows:
        raise SystemExit("No rows were generated.")

    output_csv = Path(args.output_csv).resolve()
    output_summary = Path(args.output_summary).resolve()
    write_rows(all_rows, output_csv)

    summary_rows = build_summary(all_rows)
    write_summary(summary_rows, output_summary)
    print_average_console_summary(summary_rows)
    print(f"\nSaved per-batch results to: {output_csv}")
    print(f"Saved summary results to: {output_summary}")
    print("\nAssumption used: theta is approximately inversely proportional to CPU quota for the same client.")


if __name__ == "__main__":
    main()
