from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    """Two-parameter logistic CDF."""
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit logistic CDFs for all client CSVs in the same batch and plot "
            "empirical CDFs with fitted curves on one figure."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["/home/xiaoyan/FedCompass/examples/output/result_Client1.csv"],
        help=(
            "CSV files or glob patterns used to identify one or more batches. "
            "If a file like result_Client1_5.csv is provided, the script loads "
            "all result_Client*_5.csv files from the same directory."
        ),
    )
    parser.add_argument(
        "--output-plot",
        default="examples/output/client_train_time_logistic_fit.png",
        help="Path to save the fitted plot.",
    )
    parser.add_argument(
        "--output-params",
        default="examples/output/client_train_time_logistic_params.csv",
        help="Path to save fitted theta/k parameters.",
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
        help="Column used to keep only training rows.",
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
        if candidate.exists() and candidate not in seen:
            files.append(candidate)
            seen.add(candidate)

    return files


def make_client_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("result_"):
        return stem[len("result_") :]
    return stem


def make_batch_client_label(path: Path) -> str:
    parsed = parse_result_csv_name(path)
    if parsed is None:
        return make_client_label(path)
    client_id, _ = parsed
    return f"Client{client_id}"


def parse_result_csv_name(path: Path) -> tuple[int, str] | None:
    match = re.fullmatch(r"result_Client(\d+)(?:_(.+))?", path.stem)
    if match is None:
        return None
    return int(match.group(1)), match.group(2) or ""


def batch_label_from_key(batch_key: str) -> str:
    return "base" if batch_key == "" else batch_key


def output_path_for_batch(output_path: Path, batch_key: str, multi_batch: bool) -> Path:
    if not multi_batch:
        return output_path.resolve()
    suffix = output_path.suffix
    stem = output_path.stem
    batch_label = batch_label_from_key(batch_key)
    return output_path.with_name(f"{stem}_{batch_label}{suffix}").resolve()


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
    for batch_key in sorted(batches, key=lambda item: (item == "", item)):
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
        for row in reader:
            if filter_column in row:
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


def fit_and_plot_batch(
    batch_key: str,
    batch_files: list[Path],
    args: argparse.Namespace,
    output_plot: Path,
    output_params: Path,
) -> None:
    params_rows: list[dict[str, float | int | str]] = []
    plt.figure(figsize=(14, 8))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(batch_files), 1)))

    plotted_clients = 0
    for color, csv_path in zip(colors, batch_files):
        client_label = make_batch_client_label(csv_path)

        try:
            times = load_training_times(
                csv_path=csv_path,
                time_column=args.time_column,
                round_column=args.round_column,
                filter_column=args.filter_column,
                train_flag=args.train_flag,
                max_time=args.max_time,
                min_round=args.min_round,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {client_label}: failed to load data ({exc})")
            continue

        if times.size == 0:
            print(f"[skip] {client_label}: no training samples found after filtering")
            continue

        try:
            theta_hat, k_hat = fit_logistic(times)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {client_label}: logistic fit failed ({exc})")
            continue

        x_emp = np.sort(times)
        y_emp = np.arange(1, len(x_emp) + 1, dtype=float) / float(len(x_emp))
        x_fit = np.linspace(float(x_emp.min()), float(x_emp.max()), 300)
        y_fit = logistic_cdf(x_fit, theta_hat, k_hat)

        plt.plot(
            x_emp,
            y_emp,
            "o",
            color=color,
            markersize=4,
            alpha=0.75,
            label=f"{client_label} empirical",
        )
        plt.plot(
            x_fit,
            y_fit,
            "-",
            color=color,
            linewidth=2.0,
            label=f"{client_label} fit",
        )

        params_rows.append(
            {
                "batch": batch_label_from_key(batch_key),
                "client": client_label,
                "num_samples": int(times.size),
                "theta": theta_hat,
                "k": k_hat,
                "time_min": float(x_emp.min()),
                "time_max": float(x_emp.max()),
                "time_mean": float(x_emp.mean()),
            }
        )
        plotted_clients += 1
        print(
            f"[Logistic Fit][batch={batch_label_from_key(batch_key)}] {client_label}: "
            f"theta = {theta_hat:.6f}, k = {k_hat:.6f}, n = {times.size}"
        )

    if plotted_clients == 0:
        raise SystemExit(f"No client data could be fitted for batch {batch_label_from_key(batch_key)}.")

    output_params.parent.mkdir(parents=True, exist_ok=True)
    params_rows.sort(key=lambda row: str(row["client"]))
    with output_params.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch",
                "client",
                "num_samples",
                "theta",
                "k",
                "time_min",
                "time_max",
                "time_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(params_rows)

    plt.xlabel("Training Time (s)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Client Training Time Logistic Fit (batch={batch_label_from_key(batch_key)})")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=200)
    plt.close()

    print(f"\nSaved plot to: {output_plot}")
    print(f"Saved parameters to: {output_params}")


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

    base_output_plot = Path(args.output_plot)
    base_output_params = Path(args.output_params)
    multi_batch = len(batch_files_map) > 1

    for batch_key, batch_files in batch_files_map.items():
        if not batch_files:
            continue
        fit_and_plot_batch(
            batch_key=batch_key,
            batch_files=batch_files,
            args=args,
            output_plot=output_path_for_batch(base_output_plot, batch_key, multi_batch),
            output_params=output_path_for_batch(base_output_params, batch_key, multi_batch),
        )


if __name__ == "__main__":
    main()
