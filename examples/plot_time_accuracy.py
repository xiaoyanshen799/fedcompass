#!/usr/bin/env python3
"""Plot elapsed time vs. validation accuracy from APPFL server outputs.

The script supports both the structured CSV output and the raw TXT server log.
It can draw a single run, repeated runs from one setting, or compare multiple
settings on the same figure.

Examples
--------
Single run:
    python examples/plot_time_accuracy.py \
        examples/output/result_server_5.csv

Multiple repeated runs from the same setting:
    python examples/plot_time_accuracy.py \
        'examples/output/result_server_[0-7].csv' \
        --aggregate both \
        --output examples/output/time_accuracy_repeat_runs.png

Compare two settings on one figure:
    python examples/plot_time_accuracy.py \
        --group 'setting_a=examples/output/result_server_[0-3].csv' \
        --group 'setting_b=examples/output/result_server_[4-7].csv' \
        --aggregate both \
        --output examples/output/compare_settings.png
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


TXT_LINE_RE = re.compile(
    r"server\]:\s+"
    r"(?P<update>\d+)\s+"
    r"(?P<elapsed>[0-9]*\.?[0-9]+)\s+"
    r"(?P<loss>[0-9]*\.?[0-9]+)\s+"
    r"(?P<accuracy>[0-9]*\.?[0-9]+)\s*$"
)

RUN_ALIAS_RE = re.compile(r"(?P<prefix>.+)_\[(?P<selector>[^\]]+)\](?P<suffix>\..+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot elapsed time vs validation accuracy from APPFL outputs."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input files or glob patterns, e.g. examples/output/result_server_5.csv",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="LABEL=PATTERN[,PATTERN2...]",
        help=(
            "Compare multiple settings in one figure. "
            "Each group uses one label and one or more file/glob patterns."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <first_input_or_group>_time_accuracy.png",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Defaults to an automatic title.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["auto", "none", "all", "mean", "best", "both"],
        default="auto",
        help=(
            "How to summarize multiple runs. "
            "'both' draws all runs faintly and the mean curve prominently."
        ),
    )
    parser.add_argument(
        "--best-metric",
        choices=["final", "peak"],
        default="final",
        help="Criterion for selecting the best run when --aggregate best is used.",
    )
    parser.add_argument(
        "--all-alpha",
        type=float,
        default=0.25,
        help="Line transparency when drawing every run.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help=(
            "Global multiplier applied to elapsed time before plotting and summary "
            "statistics. Useful when converting wall-clock time to CPU time."
        ),
    )
    parser.add_argument(
        "--group-time-scale",
        action="append",
        default=[],
        metavar="LABEL=FACTOR",
        help=(
            "Per-group elapsed-time multiplier applied on top of --time-scale. "
            "Example: --group-time-scale setting_a=0.45"
        ),
    )
    parser.add_argument(
        "--x-label",
        default=None,
        help="Optional custom x-axis label.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI.",
    )
    return parser.parse_args()


def parse_run_selector(selector: str) -> set[str]:
    """Expand selectors like 0-3,9,12-14 into a set of run ids."""
    run_ids: set[str] = set()
    for raw_part in selector.split(","):
        part = raw_part.strip()
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)", part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start > end:
                raise ValueError(f"Invalid descending run range '{part}' in selector '{selector}'")
            for run_id in range(start, end + 1):
                run_ids.add(str(run_id))
            continue
        if not re.fullmatch(r"\d+", part):
            raise ValueError(
                f"Invalid run selector '{part}' in '{selector}'. "
                "Use integers or inclusive ranges like 0-3,9,12-14."
            )
        run_ids.add(str(int(part)))
    if not run_ids:
        raise ValueError(f"Empty run selector '{selector}'")
    return run_ids


def resolve_appfl_run_aliases(pattern: str) -> list[Path] | None:
    """Resolve APPFL run-index patterns where run 0 is the unsuffixed file."""
    path_pattern = Path(pattern)
    parent = path_pattern.parent if str(path_pattern.parent) not in {"", "."} else Path(".")
    filename = path_pattern.name
    match = RUN_ALIAS_RE.fullmatch(filename)
    if not match or not parent.exists():
        return None

    prefix = match.group("prefix")
    selector = match.group("selector")
    suffix = match.group("suffix")
    selected_run_ids = parse_run_selector(selector)
    candidate_pattern = f"{prefix}*{suffix}"
    filename_re = re.compile(
        rf"^{re.escape(prefix)}(?:_(?P<run>\d+))?{re.escape(suffix)}$"
    )

    matched_files: list[Path] = []
    for candidate in sorted(parent.glob(candidate_pattern)):
        candidate_match = filename_re.fullmatch(candidate.name)
        if not candidate_match:
            continue
        run_id = candidate_match.group("run") or "0"
        if run_id in selected_run_ids:
            matched_files.append(candidate)
    return matched_files


def resolve_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        alias_matches = resolve_appfl_run_aliases(pattern)
        if alias_matches is not None:
            files.extend(alias_matches)
        else:
            matches = sorted(Path(p) for p in glob.glob(pattern))
            if matches:
                files.extend(matches)
            candidate = Path(pattern)
            if candidate.exists():
                files.append(candidate)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    if not deduped:
        raise FileNotFoundError("No input files matched the provided paths/patterns.")
    return deduped


def parse_group_spec(spec: str) -> tuple[str, list[str]]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid --group value '{spec}'. Expected LABEL=PATTERN[,PATTERN2...]"
        )
    label, raw_patterns = spec.split("=", 1)
    label = label.strip()
    patterns = [part.strip() for part in raw_patterns.split(",") if part.strip()]
    if not label or not patterns:
        raise ValueError(
            f"Invalid --group value '{spec}'. Expected LABEL=PATTERN[,PATTERN2...]"
        )
    return label, patterns


def parse_group_scale_spec(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise ValueError(f"Invalid --group-time-scale value '{spec}'. Expected LABEL=FACTOR")
    label, raw_factor = spec.split("=", 1)
    label = label.strip()
    raw_factor = raw_factor.strip()
    if not label or not raw_factor:
        raise ValueError(f"Invalid --group-time-scale value '{spec}'. Expected LABEL=FACTOR")
    return label, float(raw_factor)


def build_groups(
    inputs: list[str], group_specs: list[str]
) -> tuple[list[dict[str, object]], bool]:
    if inputs and group_specs:
        raise ValueError("Use positional inputs or --group, not both in the same command.")
    if not inputs and not group_specs:
        raise ValueError("Provide input files/patterns or at least one --group.")

    if group_specs:
        groups: list[dict[str, object]] = []
        for spec in group_specs:
            label, patterns = parse_group_spec(spec)
            files = resolve_inputs(patterns)
            groups.append({"label": label, "files": files})
        return groups, False

    files = resolve_inputs(inputs)
    return [{"label": "run", "files": files}], True


def build_group_time_scales(group_scale_specs: list[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for spec in group_scale_specs:
        label, factor = parse_group_scale_spec(spec)
        scales[label] = factor
    return scales


def load_run(path: Path) -> list[dict[str, float]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_run_from_csv(path)
    if suffix == ".txt":
        return load_run_from_txt(path)

    # If the extension is unusual, try CSV first, then TXT parsing.
    try:
        return load_run_from_csv(path)
    except Exception:
        return load_run_from_txt(path)


def load_run_from_csv(path: Path) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"Global Update", "Elapsed Time", "Val Accuracy"}
        if not expected.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} is missing expected CSV columns: {sorted(expected)}")
        for row in reader:
            points.append(
                {
                    "update": float(row["Global Update"]),
                    "elapsed_time": float(row["Elapsed Time"]),
                    "val_accuracy": float(row["Val Accuracy"]),
                }
            )
    if not points:
        raise ValueError(f"No data rows found in {path}")
    return points


def load_run_from_txt(path: Path) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = TXT_LINE_RE.search(line)
            if not match:
                continue
            points.append(
                {
                    "update": float(match.group("update")),
                    "elapsed_time": float(match.group("elapsed")),
                    "val_accuracy": float(match.group("accuracy")),
                }
            )
    if not points:
        raise ValueError(f"No metric rows parsed from {path}")
    return points


def scale_run_time(run: list[dict[str, float]], factor: float) -> list[dict[str, float]]:
    if factor == 1.0:
        return [dict(point) for point in run]
    return [
        {
            **point,
            "elapsed_time": float(point["elapsed_time"]) * factor,
        }
        for point in run
    ]


def apply_time_scales(
    groups: list[dict[str, object]],
    global_scale: float,
    group_scales: dict[str, float],
) -> None:
    known_labels = {str(group["label"]) for group in groups}
    unknown_labels = sorted(set(group_scales) - known_labels)
    if unknown_labels:
        raise ValueError(
            "Unknown labels in --group-time-scale: " + ", ".join(unknown_labels)
        )

    for group in groups:
        label = str(group["label"])
        runs = group["runs"]
        assert isinstance(runs, list)
        group_scale = group_scales.get(label, 1.0)
        total_scale = global_scale * group_scale
        group["time_scale"] = total_scale
        group["runs"] = [scale_run_time(run, total_scale) for run in runs]


def summarize_mean_curve(
    runs: list[list[dict[str, float]]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    by_update: dict[int, list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        for point in run:
            by_update[int(point["update"])].append(point)

    updates = sorted(by_update)
    mean_time: list[float] = []
    mean_acc: list[float] = []
    lower_acc: list[float] = []
    upper_acc: list[float] = []
    for update in updates:
        points = by_update[update]
        elapsed_values = [p["elapsed_time"] for p in points]
        acc_values = [p["val_accuracy"] for p in points]
        acc_mean = mean(acc_values)
        acc_std = pstdev(acc_values) if len(acc_values) > 1 else 0.0
        mean_time.append(mean(elapsed_values))
        mean_acc.append(acc_mean)
        lower_acc.append(acc_mean - acc_std)
        upper_acc.append(acc_mean + acc_std)
    return mean_time, mean_acc, lower_acc, upper_acc


def clip_accuracy_bounds(
    lower: list[float], upper: list[float], minimum: float = 0.0, maximum: float = 100.0
) -> tuple[list[float], list[float]]:
    """Keep plotted uncertainty bands inside the valid percentage range."""
    clipped_lower = [max(minimum, value) for value in lower]
    clipped_upper = [min(maximum, value) for value in upper]
    return clipped_lower, clipped_upper


def all_accuracies_within_percent_range(groups: list[dict[str, object]]) -> bool:
    for group in groups:
        runs = group["runs"]
        assert isinstance(runs, list)
        for run in runs:
            for point in run:
                value = float(point["val_accuracy"])
                if value < 0.0 or value > 100.0:
                    return False
    return True


def time_to_accuracy(
    run: list[dict[str, float]], threshold: float
) -> float | None:
    """Return the first elapsed time when validation accuracy reaches a threshold."""
    for point in run:
        if point["val_accuracy"] >= threshold:
            return point["elapsed_time"]
    return None


def pick_best_run(
    runs: list[list[dict[str, float]]], metric: str
) -> tuple[int, list[dict[str, float]]]:
    if metric == "peak":
        scored = [
            (idx, max(point["val_accuracy"] for point in run), run)
            for idx, run in enumerate(runs)
        ]
    else:
        scored = [
            (idx, run[-1]["val_accuracy"], run)
            for idx, run in enumerate(runs)
        ]
    best_idx, _, best_run = max(scored, key=lambda item: item[1])
    return best_idx, best_run


def infer_output_path(first_input: Path) -> Path:
    stem = first_input.stem
    return first_input.with_name(f"{stem}_time_accuracy.png")


def infer_title(groups: list[dict[str, object]], aggregate: str) -> str:
    if len(groups) == 1:
        files = groups[0]["files"]
        assert isinstance(files, list)
        if len(files) == 1:
            return f"Time vs Accuracy: {files[0].name}"
        if aggregate in {"mean", "both"}:
            return f"Time vs Accuracy ({len(files)} runs)"
        return f"Time vs Accuracy: {len(files)} runs"
    return f"Time vs Accuracy Comparison ({len(groups)} settings)"


def normalize_aggregate_mode(mode: str, groups: list[dict[str, object]]) -> str:
    if mode != "auto":
        return mode
    max_group_size = max(len(group["files"]) for group in groups)
    return "none" if max_group_size == 1 else "both"


def plot_group(
    ax: plt.Axes,
    label: str,
    files: list[Path],
    runs: list[list[dict[str, float]]],
    color: str,
    aggregate: str,
    best_metric: str,
    all_alpha: float,
    label_individual_runs: bool,
) -> None:
    if aggregate in {"none", "all", "both"}:
        for path, run in zip(files, runs):
            times = [point["elapsed_time"] for point in run]
            accuracies = [point["val_accuracy"] for point in run]
            run_label = None
            if label_individual_runs:
                run_label = path.stem if len(files) > 1 else label
            elif path == files[0]:
                run_label = f"{label} runs" if len(files) > 1 else label
            alpha = 0.95 if aggregate == "none" else all_alpha
            linewidth = 1.8 if aggregate == "none" else 1.2
            ax.plot(
                times,
                accuracies,
                linewidth=linewidth,
                alpha=alpha,
                color=color,
                label=run_label,
            )

    if aggregate in {"mean", "both"}:
        mean_time, mean_acc, lower_acc, upper_acc = summarize_mean_curve(runs)
        lower_acc, upper_acc = clip_accuracy_bounds(lower_acc, upper_acc)
        ax.plot(
            mean_time,
            mean_acc,
            color=color,
            linewidth=2.4,
            label="mean" if label_individual_runs else label,
            zorder=5,
        )
        ax.fill_between(
            mean_time,
            lower_acc,
            upper_acc,
            color=color,
            alpha=0.2,
            edgecolor="none",
            linewidth=0.0,
            label="mean ± 1 std" if label_individual_runs else None,
            zorder=4,
        )

    if aggregate == "best":
        best_idx, best_run = pick_best_run(runs, metric=best_metric)
        times = [point["elapsed_time"] for point in best_run]
        accuracies = [point["val_accuracy"] for point in best_run]
        ax.plot(
            times,
            accuracies,
            color=color,
            linewidth=2.4,
            label=(
                f"best ({files[best_idx].stem})"
                if label_individual_runs
                else f"{label} best"
            ),
        )


def plot_runs(
    groups: list[dict[str, object]],
    output: Path,
    title: str,
    aggregate: str,
    best_metric: str,
    all_alpha: float,
    dpi: int,
    label_individual_runs: bool,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, group in enumerate(groups):
        label = group["label"]
        files = group["files"]
        runs = group["runs"]
        assert isinstance(label, str)
        assert isinstance(files, list)
        assert isinstance(runs, list)
        plot_group(
            ax=ax,
            label=label,
            files=files,
            runs=runs,
            color=colors[idx % len(colors)],
            aggregate=aggregate,
            best_metric=best_metric,
            all_alpha=all_alpha,
            label_individual_runs=label_individual_runs,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation Accuracy (%)")
    if all_accuracies_within_percent_range(groups):
        ax.set_ylim(0.0, 100.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def format_summary_cell(values: list[float], total_runs: int) -> str:
    """Format a threshold summary cell with the mean and reach count."""
    if not values:
        return f"N/A (0/{total_runs})"
    return f"{mean(values):.2f} ({len(values)}/{total_runs})"


def print_group_summary(groups: list[dict[str, object]]) -> None:
    """Print aggregate metrics for each setting."""
    headers = ["Setting", "Time to 90%", "Time to 95%", "Best accuracy"]
    rows: list[list[str]] = []

    for group in groups:
        label = group["label"]
        runs = group["runs"]
        assert isinstance(label, str)
        assert isinstance(runs, list)

        tta_90 = [value for run in runs if (value := time_to_accuracy(run, 90.0)) is not None]
        tta_95 = [value for run in runs if (value := time_to_accuracy(run, 95.0)) is not None]
        best_accuracies = [max(point["val_accuracy"] for point in run) for run in runs]

        rows.append(
            [
                label,
                format_summary_cell(tta_90, len(runs)),
                format_summary_cell(tta_95, len(runs)),
                f"{mean(best_accuracies):.4f}",
            ]
        )

    widths = [
        max(len(row[idx]) for row in [headers] + rows)
        for idx in range(len(headers))
    ]

    print("\nSummary (mean across runs)")
    print("  " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("  " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    groups, label_individual_runs = build_groups(args.inputs, args.group)
    for group in groups:
        files = group["files"]
        assert isinstance(files, list)
        group["runs"] = [load_run(path) for path in files]
    apply_time_scales(
        groups=groups,
        global_scale=args.time_scale,
        group_scales=build_group_time_scales(args.group_time_scale),
    )

    aggregate = normalize_aggregate_mode(args.aggregate, groups)
    first_group_files = groups[0]["files"]
    assert isinstance(first_group_files, list)
    output = args.output or infer_output_path(first_group_files[0])
    title = args.title or infer_title(groups, aggregate)
    any_scaled = any(abs(float(group.get("time_scale", 1.0)) - 1.0) > 1e-12 for group in groups)
    x_label = args.x_label or ("CPU Time (s)" if any_scaled else "Elapsed Time (s)")

    plot_runs(
        groups=groups,
        output=output,
        title=title,
        aggregate=aggregate,
        best_metric=args.best_metric,
        all_alpha=args.all_alpha,
        dpi=args.dpi,
        label_individual_runs=label_individual_runs,
        x_label=x_label,
    )

    print(f"Saved figure to: {output}")
    if any_scaled:
        print("Applied time scaling:")
        for group in groups:
            print(f"  {group['label']}: x {float(group.get('time_scale', 1.0)):.6f}")
    print_group_summary(groups)
    for group in groups:
        label = group["label"]
        runs = group["runs"]
        assert isinstance(label, str)
        assert isinstance(runs, list)
        final_accuracies = [run[-1]["val_accuracy"] for run in runs]
        peak_accuracies = [max(point["val_accuracy"] for point in run) for run in runs]
        print(f"Group '{label}': loaded {len(runs)} run(s)")
        print(
            "  Final accuracy: "
            f"mean={mean(final_accuracies):.4f}, "
            f"best={max(final_accuracies):.4f}"
        )
        print(
            "  Peak accuracy: "
            f"mean={mean(peak_accuracies):.4f}, "
            f"best={max(peak_accuracies):.4f}"
        )


if __name__ == "__main__":
    main()
