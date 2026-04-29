from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-client CPU quotas from a completed run by measuring seconds-per-step "
            "for each client and rebalancing quotas to equalize the observed speed."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing result_seed*_Client*.csv and client_launch_manifest_*.csv.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults to the only manifest CSV inside run_dir.",
    )
    parser.add_argument(
        "--min-round",
        type=int,
        default=1,
        help="Minimum round index to include when averaging client speeds. Default drops round 0.",
    )
    parser.add_argument(
        "--preserve-total-cpu",
        action="store_true",
        default=True,
        help="Keep the total CPU budget unchanged. Enabled by default.",
    )
    parser.add_argument(
        "--min-cpu",
        type=int,
        default=1,
        help="Minimum recommended CPU quota percent.",
    )
    parser.add_argument(
        "--max-cpu",
        type=int,
        default=100,
        help="Maximum recommended CPU quota percent.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Defaults to <run_dir>/calibrated_cpu_quotas.csv.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Defaults to <run_dir>/calibrated_cpu_quotas.json.",
    )
    return parser.parse_args()


def parse_percent(value: str) -> float:
    text = value.strip()
    if text.endswith("%"):
        text = text[:-1]
    return float(text)


def format_percent(value: float) -> str:
    rounded = round(value, 4)
    if math.isclose(rounded, round(rounded), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(rounded))}%"
    return f"{rounded:.4f}".rstrip("0").rstrip(".") + "%"


def detect_manifest(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("client_launch_manifest*.csv"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one manifest CSV in {run_dir}, found {len(matches)}. "
            "Pass --manifest explicitly."
        )
    return matches[0]


def load_manifest(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"client_id", "cpu_quota"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid manifest format: {manifest_path}")
        for row in reader:
            rows.append(
                {
                    "client_id": int(str(row["client_id"]).strip()),
                    "cpu_quota": parse_percent(str(row["cpu_quota"])),
                }
            )
    rows.sort(key=lambda item: int(item["client_id"]))
    return rows


def detect_result_csv(run_dir: Path, client_id: int) -> Path:
    matches = sorted(run_dir.glob(f"*Client{client_id}.csv"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one result CSV for Client{client_id} in {run_dir}, found {len(matches)}."
        )
    return matches[0]


def load_seconds_per_step(csv_path: Path, min_round: int) -> list[float]:
    values: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Round", "Q", "Time"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid result CSV format: {csv_path}")
        for row in reader:
            round_idx = int(float(str(row["Round"]).strip()))
            if round_idx < min_round:
                continue
            local_steps = float(str(row["Q"]).strip())
            elapsed = float(str(row["Time"]).strip())
            if local_steps <= 0.0 or elapsed <= 0.0:
                continue
            values.append(elapsed / local_steps)
    if not values:
        raise SystemExit(f"No usable rows found in {csv_path} after applying --min-round={min_round}.")
    return values


def rebalance_quotas(
    rows: list[dict[str, object]],
    preserve_total_cpu: bool,
    min_cpu: int,
    max_cpu: int,
) -> list[dict[str, object]]:
    if min_cpu < 1 or max_cpu < min_cpu:
        raise ValueError("Invalid CPU bounds.")

    if preserve_total_cpu:
        total_cpu = sum(float(row["current_cpu_quota"]) for row in rows)
        raw_values = [
            float(row["current_cpu_quota"]) * float(row["avg_seconds_per_step"])
            for row in rows
        ]
        scale = total_cpu / sum(raw_values)
        recommended = [scale * value for value in raw_values]
    else:
        target = mean(float(row["avg_seconds_per_step"]) for row in rows)
        recommended = [
            float(row["current_cpu_quota"]) * float(row["avg_seconds_per_step"]) / target
            for row in rows
        ]

    bounded = [min(max(value, float(min_cpu)), float(max_cpu)) for value in recommended]
    integer = [int(math.floor(value)) for value in bounded]
    remainder = int(round(sum(bounded))) - sum(integer)

    ranked = sorted(
        enumerate(bounded),
        key=lambda item: (item[1] - math.floor(item[1]), -int(rows[item[0]]["client_id"])),
        reverse=True,
    )
    for idx, _ in ranked:
        if remainder <= 0:
            break
        if integer[idx] < max_cpu:
            integer[idx] += 1
            remainder -= 1

    for row, recommended_value, integer_value in zip(rows, bounded, integer):
        row["recommended_cpu_quota"] = recommended_value
        row["recommended_cpu_quota_rounded"] = integer_value
        row["recommended_cpu_quota_text"] = format_percent(recommended_value)
        row["recommended_cpu_quota_rounded_text"] = f"{integer_value}%"
    return rows


def build_rows(run_dir: Path, manifest_rows: list[dict[str, object]], min_round: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        client_id = int(manifest_row["client_id"])
        result_csv = detect_result_csv(run_dir, client_id)
        sps_values = load_seconds_per_step(result_csv, min_round)
        avg_sps = mean(sps_values)
        med_sps = median(sps_values)
        rows.append(
            {
                "client_id": client_id,
                "result_csv": str(result_csv),
                "current_cpu_quota": float(manifest_row["cpu_quota"]),
                "avg_seconds_per_step": avg_sps,
                "median_seconds_per_step": med_sps,
                "avg_speed": 1.0 / avg_sps,
            }
        )
    rows.sort(key=lambda item: int(item["client_id"]))
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client_id",
                "result_csv",
                "current_cpu_quota",
                "avg_seconds_per_step",
                "median_seconds_per_step",
                "avg_speed",
                "recommended_cpu_quota",
                "recommended_cpu_quota_text",
                "recommended_cpu_quota_rounded",
                "recommended_cpu_quota_rounded_text",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, object]], output_path: Path, run_dir: Path, manifest_path: Path, min_round: int) -> None:
    payload = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "min_round": min_round,
        "recommended_cpu_quota_csv": ",".join(
            str(row["recommended_cpu_quota_rounded_text"]) for row in rows
        ),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else detect_manifest(run_dir)
    output_csv = Path(args.output_csv).resolve() if args.output_csv else run_dir / "calibrated_cpu_quotas.csv"
    output_json = Path(args.output_json).resolve() if args.output_json else run_dir / "calibrated_cpu_quotas.json"

    manifest_rows = load_manifest(manifest_path)
    rows = build_rows(run_dir, manifest_rows, args.min_round)
    rows = rebalance_quotas(
        rows=rows,
        preserve_total_cpu=args.preserve_total_cpu,
        min_cpu=args.min_cpu,
        max_cpu=args.max_cpu,
    )
    write_csv(rows, output_csv)
    write_json(rows, output_json, run_dir, manifest_path, args.min_round)

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")
    print("")
    for row in rows:
        print(
            f"Client{int(row['client_id']):02d}: "
            f"cpu={float(row['current_cpu_quota']):.0f}% -> "
            f"{str(row['recommended_cpu_quota_rounded_text'])} "
            f"(avg_s_per_step={float(row['avg_seconds_per_step']):.6f}, "
            f"avg_speed={float(row['avg_speed']):.6f})"
        )
    print("")
    print("Recommended CPU quotas:")
    print(",".join(str(row["recommended_cpu_quota_rounded_text"]) for row in rows))


if __name__ == "__main__":
    main()
