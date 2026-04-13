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
        'examples/output/result_server_[1-7].csv' \
        --aggregate both \
        --output examples/output/time_accuracy_repeat_runs.png

Compare two settings on one figure:
    python examples/plot_time_accuracy.py \
        --group 'setting_a=examples/output/result_server_[1-3].csv' \
        --group 'setting_b=examples/output/result_server_[4-7].csv' \
        --aggregate both \
        --output examples/output/compare_settings.png
"""

from __future__ import annotations

import argparse
import csv
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
        "--dpi",
        type=int,
        default=180,
        help="Output DPI.",
    )
    return parser.parse_args()


def resolve_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if matches:
            files.extend(matches)
        else:
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

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    groups, label_individual_runs = build_groups(args.inputs, args.group)
    for group in groups:
        files = group["files"]
        assert isinstance(files, list)
        group["runs"] = [load_run(path) for path in files]

    aggregate = normalize_aggregate_mode(args.aggregate, groups)
    first_group_files = groups[0]["files"]
    assert isinstance(first_group_files, list)
    output = args.output or infer_output_path(first_group_files[0])
    title = args.title or infer_title(groups, aggregate)

    plot_runs(
        groups=groups,
        output=output,
        title=title,
        aggregate=aggregate,
        best_metric=args.best_metric,
        all_alpha=args.all_alpha,
        dpi=args.dpi,
        label_individual_runs=label_individual_runs,
    )

    print(f"Saved figure to: {output}")
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
