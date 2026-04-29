#!/usr/bin/env python3
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


CLIENT_FILE_RE = re.compile(r"(?P<prefix>.+)_Client(?P<client_id>\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot empirical CDFs of client training time from one run batch and fit logistic CDFs."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more CSV files or glob patterns. If one or more files like "
            "'result_seed42_fedavg_Client1.csv' are provided, the script loads all "
            "matching client CSVs from the same batch."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <batch_prefix>_client_time_cdf.png.",
    )
    parser.add_argument(
        "--time-column",
        default="Time",
        help="Column name containing per-round client time.",
    )
    parser.add_argument(
        "--round-column",
        default="Round",
        help="Optional column name used for sorting and filtering.",
    )
    parser.add_argument(
        "--min-round",
        type=float,
        default=0.0,
        help="Minimum round index to include. Default keeps all rounds.",
    )
    parser.add_argument(
        "--max-round",
        type=float,
        default=None,
        help="Optional maximum round index to include.",
    )
    parser.add_argument(
        "--first-n-rounds",
        type=int,
        default=None,
        help=(
            "Keep only the first N rounds for the empirical plot. "
            "For example, --first-n-rounds 30 keeps rounds 0-29 by default."
        ),
    )
    parser.add_argument(
        "--fit-min-round",
        type=float,
        default=None,
        help=(
            "Minimum round index used only for logistic fitting. "
            "Defaults to --min-round when omitted."
        ),
    )
    parser.add_argument(
        "--fit-max-round",
        type=float,
        default=None,
        help=(
            "Maximum round index used only for logistic fitting. "
            "Example: --fit-max-round 49 fits only the first 50 rounds when rounds start at 0."
        ),
    )
    parser.add_argument(
        "--fit-first-n-rounds",
        type=int,
        default=None,
        help=(
            "Fit the logistic CDF using only the first N rounds from the fitting range. "
            "For example, --fit-first-n-rounds 50 keeps rounds 0-49 by default."
        ),
    )
    parser.add_argument(
        "--output-params",
        type=Path,
        default=None,
        help="Optional CSV path to save per-client logistic fit parameters.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--x-label",
        default="Client Time (s)",
        help="X-axis label.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI.",
    )
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        expanded_pattern = str(Path(pattern).expanduser())
        matches = sorted(Path(match).resolve() for match in glob.glob(expanded_pattern))
        if matches:
            for match in matches:
                if match.is_file() and match not in seen:
                    files.append(match)
                    seen.add(match)
            continue

        candidate = Path(pattern).expanduser().resolve()
        if candidate.exists() and candidate.is_file() and candidate not in seen:
            files.append(candidate)
            seen.add(candidate)

    if not files:
        raise FileNotFoundError("No input CSV files matched the provided paths or patterns.")
    return files


def parse_client_file(path: Path) -> tuple[str, int] | None:
    match = CLIENT_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("client_id"))


def collect_batch_files(paths: list[Path]) -> tuple[str, list[Path]]:
    parsed_paths: list[tuple[Path, str, int]] = []
    for path in paths:
        parsed = parse_client_file(path)
        if parsed is None:
            continue
        prefix, client_id = parsed
        parsed_paths.append((path, prefix, client_id))

    if not parsed_paths:
        raise ValueError(
            "No client CSVs found. Expected filenames like result_seed42_fedavg_Client1.csv."
        )

    prefixes = {prefix for _, prefix, _ in parsed_paths}
    if len(prefixes) != 1:
        raise ValueError(
            "Inputs span multiple batches. Pass files from exactly one batch, or use a single glob "
            "such as 'examples/exponential-10-fedavg-byclass/result_seed42_fedavg_Client*.csv'."
        )

    batch_prefix = parsed_paths[0][1]
    parent_dir = parsed_paths[0][0].parent
    batch_pairs: list[tuple[int, Path]] = []
    for match in parent_dir.glob(f"{batch_prefix}_Client*.csv"):
        parsed_match = parse_client_file(match)
        if parsed_match is None:
            continue
        _, client_id = parsed_match
        batch_pairs.append((client_id, match.resolve()))
    batch_files = [path for client_id, path in sorted(batch_pairs, key=lambda item: item[0])]
    if not batch_files:
        raise ValueError(f"No client CSVs found for batch prefix '{batch_prefix}'.")
    return batch_prefix, batch_files


def infer_output_path(batch_files: list[Path], batch_prefix: str) -> Path:
    return batch_files[0].parent / f"{batch_prefix}_client_time_cdf.png"


def make_client_label(path: Path) -> str:
    parsed = parse_client_file(path)
    if parsed is None:
        return path.stem
    _, client_id = parsed
    return f"Client{client_id}"


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    """Two-parameter logistic CDF."""
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def load_client_records(
    csv_path: Path,
    time_column: str,
    round_column: str,
    min_round: float | None,
    max_round: float | None,
) -> list[tuple[float, float]]:
    records: list[tuple[float, float]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or time_column not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing required column '{time_column}'.")

        for row in reader:
            raw_time = str(row.get(time_column, "")).strip()
            if not raw_time:
                continue

            try:
                time_value = float(raw_time)
            except ValueError:
                continue

            round_value = float("inf")
            if round_column in row:
                raw_round = str(row.get(round_column, "")).strip()
                if raw_round:
                    try:
                        round_value = float(raw_round)
                    except ValueError:
                        round_value = float("inf")

            if min_round is not None and round_value < min_round:
                continue
            if max_round is not None and round_value > max_round:
                continue

            records.append((round_value, time_value))

    records.sort(key=lambda item: (item[0], item[1]))
    return records


def records_to_times(records: list[tuple[float, float]]) -> np.ndarray:
    if not records:
        return np.array([], dtype=float)
    return np.array([time_value for _, time_value in records], dtype=float)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.astype(float))
    # Use midpoint plotting positions so the first and last samples stay
    # strictly inside (0, 1). This makes the empirical points visually align
    # better with fitted continuous CDFs.
    y = (np.arange(1, len(x) + 1, dtype=float) - 0.5) / float(len(x))
    return x, y


def logistic_fit_loss(
    x_data: np.ndarray,
    y_data: np.ndarray,
    theta: float,
    k: float,
) -> float:
    pred = logistic_cdf(x_data, theta, k)
    return float(np.mean((pred - y_data) ** 2))


def fit_logistic(times: np.ndarray) -> tuple[float, float, float]:
    x_data, y_data = empirical_cdf(times)
    if x_data.size < 2:
        raise ValueError("At least two samples are required for logistic fitting.")

    q25, q50, q75 = np.quantile(x_data, [0.25, 0.5, 0.75])
    data_range = max(float(x_data.max() - x_data.min()), 1e-6)
    iqr = max(float(q75 - q25), 1e-6)

    # Logistic IQR = 2 * k * ln(3), so this is a robust initial guess.
    best_theta = float(q50)
    best_k = max(iqr / (2.0 * float(np.log(3.0))), data_range / 50.0, 1e-3)
    best_loss = logistic_fit_loss(x_data, y_data, best_theta, best_k)

    theta_step = max(iqr, data_range / 4.0, 1e-3)
    k_step = max(best_k, data_range / 6.0, 1e-3)

    for _ in range(12):
        theta_candidates = np.linspace(best_theta - theta_step, best_theta + theta_step, 9)
        k_min = max(best_k - k_step, 1e-4)
        k_max = max(best_k + k_step, k_min * 1.01)
        k_candidates = np.linspace(k_min, k_max, 9)

        improved = False
        for theta_candidate in theta_candidates:
            for k_candidate in k_candidates:
                loss = logistic_fit_loss(x_data, y_data, float(theta_candidate), float(k_candidate))
                if loss < best_loss:
                    best_theta = float(theta_candidate)
                    best_k = float(k_candidate)
                    best_loss = float(loss)
                    improved = True

        theta_step *= 0.5
        k_step *= 0.5
        if not improved and max(theta_step, k_step) < 1e-4:
            break

    return best_theta, best_k, best_loss


def resolve_fit_round_window(args: argparse.Namespace) -> tuple[float, float | None]:
    fit_min_round = args.fit_min_round if args.fit_min_round is not None else args.min_round
    fit_max_round = args.fit_max_round
    if args.fit_first_n_rounds is not None:
        if args.fit_first_n_rounds <= 0:
            raise ValueError("--fit-first-n-rounds must be positive.")
        fit_max_from_count = fit_min_round + float(args.fit_first_n_rounds) - 1.0
        fit_max_round = (
            fit_max_from_count
            if fit_max_round is None
            else min(float(fit_max_round), fit_max_from_count)
        )
    return fit_min_round, fit_max_round


def resolve_plot_round_window(args: argparse.Namespace) -> tuple[float, float | None]:
    plot_min_round = args.min_round
    plot_max_round = args.max_round
    if args.first_n_rounds is not None:
        if args.first_n_rounds <= 0:
            raise ValueError("--first-n-rounds must be positive.")
        plot_max_from_count = plot_min_round + float(args.first_n_rounds) - 1.0
        plot_max_round = (
            plot_max_from_count
            if plot_max_round is None
            else min(float(plot_max_round), plot_max_from_count)
        )
    return plot_min_round, plot_max_round


def plot_batch(
    batch_prefix: str,
    batch_files: list[Path],
    output: Path,
    output_params: Path | None,
    args: argparse.Namespace,
) -> None:
    plt.figure(figsize=(12, 7))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(batch_files), 1)))
    fit_rows: list[dict[str, float | int | str]] = []
    plot_min_round, plot_max_round = resolve_plot_round_window(args)
    fit_min_round, fit_max_round = resolve_fit_round_window(args)

    plotted_clients = 0
    for color, csv_path in zip(colors, batch_files):
        client_label = make_client_label(csv_path)
        plot_records = load_client_records(
            csv_path=csv_path,
            time_column=args.time_column,
            round_column=args.round_column,
            min_round=plot_min_round,
            max_round=plot_max_round,
        )
        plot_times = records_to_times(plot_records)
        if plot_times.size == 0:
            print(f"[skip] {client_label}: no samples after filtering")
            continue

        fit_records = load_client_records(
            csv_path=csv_path,
            time_column=args.time_column,
            round_column=args.round_column,
            min_round=fit_min_round,
            max_round=fit_max_round,
        )
        fit_times = records_to_times(fit_records)
        if fit_times.size == 0:
            print(f"[skip] {client_label}: no samples available for logistic fitting")
            continue

        x, y = empirical_cdf(plot_times)
        plt.plot(
            x,
            y,
            "o",
            color=color,
            markersize=4,
            alpha=0.7,
            label=f"{client_label} empirical",
        )

        theta_hat, k_hat, fit_loss = fit_logistic(fit_times)
        fit_x_min = min(float(plot_times.min()), float(fit_times.min()), theta_hat - 6.0 * k_hat)
        fit_x_max = max(float(plot_times.max()), float(fit_times.max()), theta_hat + 6.0 * k_hat)
        fit_x = np.linspace(fit_x_min, fit_x_max, 400)
        fit_y = logistic_cdf(fit_x, theta_hat, k_hat)
        plt.plot(
            fit_x,
            fit_y,
            "--",
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=f"{client_label} logistic",
        )

        fit_rows.append(
            {
                "client": client_label,
                "num_plot_samples": int(plot_times.size),
                "num_fit_samples": int(fit_times.size),
                "plot_min_round": plot_min_round,
                "plot_max_round": "" if plot_max_round is None else float(plot_max_round),
                "fit_min_round": fit_min_round,
                "fit_max_round": "" if fit_max_round is None else float(fit_max_round),
                "theta": theta_hat,
                "k": k_hat,
                "fit_mse": fit_loss,
                "time_min": float(fit_times.min()),
                "time_mean": float(fit_times.mean()),
                "time_max": float(fit_times.max()),
            }
        )
        plotted_clients += 1
        print(
            f"[CDF] {client_label}: plot_rounds = [{plot_min_round}, "
            f"{'end' if plot_max_round is None else int(plot_max_round)}], "
            f"n = {plot_times.size}, min = {plot_times.min():.4f}, "
            f"mean = {plot_times.mean():.4f}, max = {plot_times.max():.4f}"
        )
        print(
            f"[Logistic Fit] {client_label}: fit_rounds = [{fit_min_round}, "
            f"{'end' if fit_max_round is None else int(fit_max_round)}], "
            f"fit_n = {fit_times.size}, "
            f"theta = {theta_hat:.4f}, k = {k_hat:.4f}, mse = {fit_loss:.6f}"
        )

    if plotted_clients == 0:
        raise SystemExit("No client time series available after filtering.")

    if output_params is not None:
        output_params.parent.mkdir(parents=True, exist_ok=True)
        with output_params.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "client",
                    "num_plot_samples",
                    "num_fit_samples",
                    "plot_min_round",
                    "plot_max_round",
                    "fit_min_round",
                    "fit_max_round",
                    "theta",
                    "k",
                    "fit_mse",
                    "time_min",
                    "time_mean",
                    "time_max",
                ],
            )
            writer.writeheader()
            writer.writerows(fit_rows)
        print(f"Saved fit parameters to: {output_params}")

    title = args.title or f"Client Time CDF ({batch_prefix})"
    plt.xlabel(args.x_label)
    plt.ylabel("Cumulative Probability")
    plt.title(title)
    plt.xlim(left=0.0)
    plt.ylim(0.0, 1.01)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=args.dpi)
    plt.close()

    print(f"\nSaved plot to: {output}")


def main() -> None:
    args = parse_args()
    input_files = expand_inputs(args.inputs)
    batch_prefix, batch_files = collect_batch_files(input_files)
    output = args.output.resolve() if args.output is not None else infer_output_path(batch_files, batch_prefix)
    output_params = args.output_params.resolve() if args.output_params is not None else None
    plot_batch(
        batch_prefix=batch_prefix,
        batch_files=batch_files,
        output=output,
        output_params=output_params,
        args=args,
    )


if __name__ == "__main__":
    main()
