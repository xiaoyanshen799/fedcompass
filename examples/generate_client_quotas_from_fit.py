from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate client target timings/speeds from fitted theta(CPU) curves, "
            "then convert them into per-client CPU quotas."
        )
    )
    parser.add_argument(
        "--fit-csv",
        default="examples/output/cpu_theta_runs/concurrent_quota_theta_fit.csv",
        help="Path to concurrent_quota_theta_fit.csv.",
    )
    parser.add_argument(
        "--distribution",
        choices=["homogeneous", "homogenous", "normal", "exponential"],
        default="normal",
        help="Target client distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--max-cpu",
        type=float,
        default=100.0,
        help="Upper bound for generated CPU quota. 100 means one full core.",
    )
    parser.add_argument(
        "--min-cpu",
        type=float,
        default=1.0,
        help="Lower bound for generated CPU quota.",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.98,
        help=(
            "Safety factor for the fastest feasible target. "
            "For theta-space sampling, the minimum feasible theta is theta(max_cpu)/headroom."
        ),
    )
    parser.add_argument(
        "--assignment",
        choices=["sorted", "random"],
        default="sorted",
        help=(
            "How to assign generated targets to clients. "
            "'sorted' matches faster targets to stronger fitted clients."
        ),
    )
    parser.add_argument(
        "--normal-space",
        choices=["theta", "speed"],
        default="theta",
        help=(
            "For normal distribution, sample in theta space or speed space. "
            "Default is theta."
        ),
    )
    parser.add_argument(
        "--normal-mean",
        type=float,
        default=1.0,
        help="Mean of the raw normal samples when sampling in speed space.",
    )
    parser.add_argument(
        "--normal-std",
        type=float,
        default=0.15,
        help="Std of the raw normal samples when sampling in speed space.",
    )
    parser.add_argument(
        "--normal-theta-mean",
        type=float,
        default=None,
        help=(
            "Mean theta for normal(theta) sampling. If omitted, use the average theta "
            "at --normal-theta-mean-from-cpu."
        ),
    )
    parser.add_argument(
        "--normal-theta-mean-from-cpu",
        type=float,
        default=50.0,
        help="Reference CPU quota used to derive the default theta mean.",
    )
    parser.add_argument(
        "--normal-theta-std",
        type=float,
        default=None,
        help="Std theta for normal(theta) sampling. If omitted, use --normal-theta-std-ratio * mean_theta.",
    )
    parser.add_argument(
        "--normal-theta-std-ratio",
        type=float,
        default=0.3,
        help="Std ratio used when --normal-theta-std is omitted.",
    )
    parser.add_argument(
        "--exp-scale",
        type=float,
        default=1.0,
        help="Scale of the raw exponential samples when sampling in speed space.",
    )
    parser.add_argument(
        "--output-csv",
        default="examples/output/generated_client_cpu_quotas.csv",
        help="Path to save the generated target speeds/thetas and CPU quotas.",
    )
    parser.add_argument(
        "--output-json",
        default="examples/output/generated_client_cpu_quotas.json",
        help="Path to save metadata for the generated quotas.",
    )
    return parser.parse_args()


def load_fit_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"client_id", "client", "fit_model", "a", "b", "p"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid fit CSV format: {path}")

        for row in reader:
            fit_model = str(row["fit_model"]).strip()
            if fit_model not in {"inverse_power", "inverse_linear"}:
                raise ValueError(
                    f"Only inverse_power and inverse_linear fits are supported. "
                    f"Client {row.get('client_id', '?')} has fit_model={fit_model!r}."
                )
            rows.append(
                {
                    "client_id": int(str(row["client_id"]).strip()),
                    "client": str(row["client"]).strip(),
                    "a": float(str(row["a"]).strip()),
                    "b": float(str(row["b"]).strip()),
                    "p": float(str(row["p"]).strip()),
                    "cpu_fit_scale": float(str(row.get("cpu_fit_scale", "1.0")).strip() or "1.0"),
                    "fit_model": fit_model,
                    "r2": float(str(row.get("r2", "nan")).strip() or "nan"),
                }
            )
    if not rows:
        raise ValueError(f"No fit rows found in {path}")
    rows.sort(key=lambda item: int(item["client_id"]))
    return rows


def theta_from_cpu(cpu: float, a: float, b: float, p: float, fit_model: str, cpu_fit_scale: float) -> float:
    cpu_input = cpu / cpu_fit_scale
    if fit_model == "inverse_linear":
        return a + b / cpu_input
    return a + b * (cpu_input ** (-p))


def speed_from_theta(theta: float) -> float:
    if theta <= 0.0:
        raise ValueError(f"Theta must be positive, got {theta}")
    return 1.0 / theta


def cpu_from_theta(theta: float, a: float, b: float, p: float, fit_model: str, cpu_fit_scale: float) -> float:
    margin = theta - a
    if margin <= 0.0:
        raise ValueError(
            f"Infeasible theta={theta:.8f} for fit a={a:.8f}, b={b:.8f}, p={p:.8f}: theta must exceed a."
        )
    if fit_model == "inverse_linear":
        return (b / margin) * cpu_fit_scale
    return ((b / margin) ** (1.0 / p)) * cpu_fit_scale


def format_percent(value: float) -> str:
    rounded = round(value, 4)
    if math.isclose(rounded, round(rounded), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(rounded))}%"
    return f"{rounded:.4f}".rstrip("0").rstrip(".") + "%"


def normalize_distribution_name(name: str) -> str:
    return "homogeneous" if name == "homogenous" else name


def build_enriched_rows(
    fit_rows: list[dict[str, Any]],
    min_cpu: float,
    max_cpu: float,
    headroom: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fit_rows:
        theta_at_max_cpu = theta_from_cpu(
            max_cpu,
            float(row["a"]),
            float(row["b"]),
            float(row["p"]),
            str(row["fit_model"]),
            float(row["cpu_fit_scale"]),
        )
        theta_at_min_cpu = theta_from_cpu(
            min_cpu,
            float(row["a"]),
            float(row["b"]),
            float(row["p"]),
            str(row["fit_model"]),
            float(row["cpu_fit_scale"]),
        )
        enriched = dict(row)
        enriched["theta_at_max_cpu"] = theta_at_max_cpu
        enriched["theta_at_min_cpu"] = theta_at_min_cpu
        enriched["theta_lower_bound"] = theta_at_max_cpu / headroom
        enriched["theta_upper_bound"] = theta_at_min_cpu
        enriched["max_feasible_speed"] = speed_from_theta(theta_at_max_cpu)
        rows.append(enriched)
    return rows


def generate_positive_normal(
    rng: np.random.Generator,
    count: int,
    mean: float,
    std: float,
) -> np.ndarray:
    values = rng.normal(loc=mean, scale=std, size=count)
    retries = 0
    while np.any(values <= 0.0):
        mask = values <= 0.0
        values[mask] = rng.normal(loc=mean, scale=std, size=int(mask.sum()))
        retries += 1
        if retries > 1000:
            raise RuntimeError("Failed to draw positive normal samples.")
    return values.astype(float)


def resolve_normal_theta_params(
    args: argparse.Namespace,
    fit_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    if args.normal_theta_mean is not None:
        mean_theta = float(args.normal_theta_mean)
    else:
        reference_cpu = float(args.normal_theta_mean_from_cpu)
        if reference_cpu <= 0.0:
            raise ValueError("--normal-theta-mean-from-cpu must be positive.")
        theta_values = [
            theta_from_cpu(
                reference_cpu,
                float(row["a"]),
                float(row["b"]),
                float(row["p"]),
                str(row["fit_model"]),
                float(row["cpu_fit_scale"]),
            )
            for row in fit_rows
        ]
        mean_theta = float(np.mean(theta_values))

    if mean_theta <= 0.0:
        raise ValueError(f"Normal theta mean must be positive, got {mean_theta}")

    if args.normal_theta_std is not None:
        std_theta = float(args.normal_theta_std)
    else:
        std_theta = float(args.normal_theta_std_ratio) * mean_theta

    if std_theta <= 0.0:
        raise ValueError(f"Normal theta std must be positive, got {std_theta}")

    return mean_theta, std_theta


def generate_raw_speed_samples(
    args: argparse.Namespace,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dist = normalize_distribution_name(args.distribution)
    if dist == "homogeneous":
        return np.ones(count, dtype=float)
    if dist == "normal":
        return generate_positive_normal(rng, count, args.normal_mean, args.normal_std)
    if dist == "exponential":
        values = rng.exponential(scale=args.exp_scale, size=count).astype(float)
        return np.maximum(values, 1e-8)
    raise ValueError(f"Unsupported distribution: {args.distribution}")


def assign_target_speeds(
    enriched_rows: list[dict[str, Any]],
    raw_speeds: np.ndarray,
    max_cpu: float,
    headroom: float,
    assignment: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if assignment == "sorted":
        rows_ordered = sorted(enriched_rows, key=lambda item: float(item["max_feasible_speed"]), reverse=True)
        speed_order = np.sort(raw_speeds)[::-1]
    elif assignment == "random":
        rows_ordered = list(enriched_rows)
        rng.shuffle(rows_ordered)
        speed_order = np.array(raw_speeds, dtype=float)
        rng.shuffle(speed_order)
    else:
        raise ValueError(f"Unsupported assignment mode: {assignment}")

    feasible_scales = []
    for row, raw_speed in zip(rows_ordered, speed_order):
        feasible_scales.append(headroom * float(row["max_feasible_speed"]) / float(raw_speed))
    global_scale = min(feasible_scales)

    assigned_rows: list[dict[str, Any]] = []
    for row, raw_speed in zip(rows_ordered, speed_order):
        target_speed = global_scale * float(raw_speed)
        target_theta = 1.0 / target_speed
        cpu_target = min(
            cpu_from_theta(
                target_theta,
                float(row["a"]),
                float(row["b"]),
                float(row["p"]),
                str(row["fit_model"]),
                float(row["cpu_fit_scale"]),
            ),
            max_cpu,
        )
        assigned = dict(row)
        assigned["raw_theta"] = float(target_theta)
        assigned["raw_speed"] = float(raw_speed)
        assigned["target_speed"] = float(target_speed)
        assigned["target_theta"] = float(target_theta)
        assigned["cpu_quota_percent"] = float(cpu_target)
        assigned_rows.append(assigned)

    assigned_rows.sort(key=lambda item: int(item["client_id"]))
    return assigned_rows


def generate_feasible_normal_theta_samples(
    rows_ordered: list[dict[str, Any]],
    mean_theta: float,
    std_theta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(rows_ordered)
    for _ in range(5000):
        theta_values = np.sort(generate_positive_normal(rng, count, mean_theta, std_theta))
        feasible = True
        for row, theta_value in zip(rows_ordered, theta_values):
            if not (float(row["theta_lower_bound"]) <= float(theta_value) <= float(row["theta_upper_bound"])):
                feasible = False
                break
        if feasible:
            return theta_values
    raise RuntimeError(
        "Failed to generate feasible normal(theta) samples. "
        "Try increasing --max-cpu, decreasing --min-cpu, or reducing --normal-theta-std."
    )


def assign_target_thetas(
    enriched_rows: list[dict[str, Any]],
    target_thetas: np.ndarray,
    max_cpu: float,
    assignment: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if assignment == "sorted":
        rows_ordered = sorted(enriched_rows, key=lambda item: float(item["theta_at_max_cpu"]))
        theta_order = np.sort(np.asarray(target_thetas, dtype=float))
    elif assignment == "random":
        rows_ordered = list(enriched_rows)
        rng.shuffle(rows_ordered)
        theta_order = np.array(target_thetas, dtype=float)
        rng.shuffle(theta_order)
    else:
        raise ValueError(f"Unsupported assignment mode: {assignment}")

    assigned_rows: list[dict[str, Any]] = []
    for row, theta_target in zip(rows_ordered, theta_order):
        cpu_target = min(
            cpu_from_theta(
                float(theta_target),
                float(row["a"]),
                float(row["b"]),
                float(row["p"]),
                str(row["fit_model"]),
                float(row["cpu_fit_scale"]),
            ),
            max_cpu,
        )
        assigned = dict(row)
        assigned["raw_theta"] = float(theta_target)
        assigned["raw_speed"] = float(speed_from_theta(float(theta_target)))
        assigned["target_speed"] = float(speed_from_theta(float(theta_target)))
        assigned["target_theta"] = float(theta_target)
        assigned["cpu_quota_percent"] = float(cpu_target)
        assigned_rows.append(assigned)

    assigned_rows.sort(key=lambda item: int(item["client_id"]))
    return assigned_rows


def generate_distribution_rows(
    args: argparse.Namespace,
    fit_rows: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    enriched_rows = build_enriched_rows(fit_rows, args.min_cpu, args.max_cpu, args.headroom)
    dist = normalize_distribution_name(args.distribution)

    if dist == "normal" and args.normal_space == "theta":
        mean_theta, std_theta = resolve_normal_theta_params(args, fit_rows)
        if args.assignment == "sorted":
            rows_for_sampling = sorted(enriched_rows, key=lambda item: float(item["theta_at_max_cpu"]))
        else:
            rows_for_sampling = list(enriched_rows)
        theta_samples = generate_feasible_normal_theta_samples(rows_for_sampling, mean_theta, std_theta, rng)
        assigned_rows = assign_target_thetas(
            enriched_rows=enriched_rows,
            target_thetas=theta_samples,
            max_cpu=args.max_cpu,
            assignment=args.assignment,
            rng=rng,
        )
        return assigned_rows, {"normal_theta_mean": mean_theta, "normal_theta_std": std_theta}

    raw_speeds = generate_raw_speed_samples(args, len(fit_rows), rng)
    assigned_rows = assign_target_speeds(
        enriched_rows=enriched_rows,
        raw_speeds=raw_speeds,
        max_cpu=args.max_cpu,
        headroom=args.headroom,
        assignment=args.assignment,
        rng=rng,
    )
    return assigned_rows, {}


def write_csv(rows: list[dict[str, Any]], output_path: Path, min_cpu: float, max_cpu: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client_id",
                "client",
                "fit_model",
                "r2",
                "gamma_a",
                "alpha_b",
                "beta_p",
                "cpu_fit_scale",
                "theta_at_max_cpu",
                "theta_at_min_cpu",
                "max_feasible_speed",
                "raw_theta",
                "raw_speed",
                "target_speed",
                "target_theta",
                "cpu_quota_percent",
                "cpu_quota_text",
            ],
        )
        writer.writeheader()
        for row in rows:
            cpu_quota_percent = min(max(float(row["cpu_quota_percent"]), min_cpu), max_cpu)
            writer.writerow(
                {
                    "client_id": int(row["client_id"]),
                    "client": str(row["client"]),
                    "fit_model": str(row["fit_model"]),
                    "r2": float(row["r2"]),
                    "gamma_a": float(row["a"]),
                    "alpha_b": float(row["b"]),
                    "beta_p": float(row["p"]),
                    "cpu_fit_scale": float(row["cpu_fit_scale"]),
                    "theta_at_max_cpu": float(row["theta_at_max_cpu"]),
                    "theta_at_min_cpu": float(row["theta_at_min_cpu"]),
                    "max_feasible_speed": float(row["max_feasible_speed"]),
                    "raw_theta": float(row["raw_theta"]),
                    "raw_speed": float(row["raw_speed"]),
                    "target_speed": float(row["target_speed"]),
                    "target_theta": float(row["target_theta"]),
                    "cpu_quota_percent": cpu_quota_percent,
                    "cpu_quota_text": format_percent(cpu_quota_percent),
                }
            )


def write_json(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    extras: dict[str, float],
    output_path: Path,
    min_cpu: float,
    max_cpu: float,
) -> None:
    cpu_values = [min(max(float(row["cpu_quota_percent"]), min_cpu), max_cpu) for row in rows]
    payload = {
        "fit_csv": str(Path(args.fit_csv).resolve()),
        "distribution": args.distribution,
        "seed": args.seed,
        "assignment": args.assignment,
        "max_cpu": args.max_cpu,
        "min_cpu": args.min_cpu,
        "headroom": args.headroom,
        "normal_space": args.normal_space,
        "normal_mean": args.normal_mean,
        "normal_std": args.normal_std,
        "normal_theta_mean": args.normal_theta_mean,
        "normal_theta_mean_from_cpu": args.normal_theta_mean_from_cpu,
        "normal_theta_std": args.normal_theta_std,
        "normal_theta_std_ratio": args.normal_theta_std_ratio,
        "exp_scale": args.exp_scale,
        **extras,
        "cpu_quota_values": cpu_values,
        "cpu_quota_texts": [format_percent(value) for value in cpu_values],
        "cpu_quota_csv": ",".join(format_percent(value) for value in cpu_values),
        "rows": [
            {
                "client_id": int(row["client_id"]),
                "client": str(row["client"]),
                "target_speed": float(row["target_speed"]),
                "target_theta": float(row["target_theta"]),
                "cpu_quota_percent": min(max(float(row["cpu_quota_percent"]), min_cpu), max_cpu),
            }
            for row in rows
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    fit_csv = Path(args.fit_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()

    if args.max_cpu <= 0.0:
        raise SystemExit("--max-cpu must be positive.")
    if args.min_cpu <= 0.0:
        raise SystemExit("--min-cpu must be positive.")
    if args.min_cpu > args.max_cpu:
        raise SystemExit("--min-cpu must be <= --max-cpu.")
    if not (0.0 < args.headroom <= 1.0):
        raise SystemExit("--headroom must be in (0, 1].")

    fit_rows = load_fit_rows(fit_csv)
    rng = np.random.default_rng(args.seed)
    assigned_rows, extras = generate_distribution_rows(args, fit_rows, rng)

    for row in assigned_rows:
        row["cpu_quota_percent"] = min(max(float(row["cpu_quota_percent"]), args.min_cpu), args.max_cpu)

    write_csv(assigned_rows, output_csv, args.min_cpu, args.max_cpu)
    write_json(assigned_rows, args, extras, output_json, args.min_cpu, args.max_cpu)

    print(f"Loaded {len(fit_rows)} client fits from: {fit_csv}")
    print(f"Distribution: {args.distribution}, seed={args.seed}, assignment={args.assignment}")
    if "normal_theta_mean" in extras:
        print(
            "Normal(theta) settings: "
            f"mu={float(extras['normal_theta_mean']):.6f}s, "
            f"sigma={float(extras['normal_theta_std']):.6f}s, "
            f"reference_cpu={args.normal_theta_mean_from_cpu}%"
        )
    print(f"Saved generated quotas to: {output_csv}")
    print(f"Saved metadata to: {output_json}")
    print("")
    print("Generated targets:")
    for row in assigned_rows:
        print(
            f"  Client{int(row['client_id']):02d}: "
            f"speed={float(row['target_speed']):.8f}, "
            f"theta={float(row['target_theta']):.6f}s, "
            f"cpu={format_percent(float(row['cpu_quota_percent']))}"
        )
    print("")
    print("CPU quotas CSV:")
    print(",".join(format_percent(float(row["cpu_quota_percent"])) for row in assigned_rows))


if __name__ == "__main__":
    main()
