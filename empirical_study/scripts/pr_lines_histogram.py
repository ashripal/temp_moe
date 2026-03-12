"""Generate histogram of modified lines (additions + deletions) from PR CSV."""

from __future__ import annotations

import argparse
import csv
import os
from typing import List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a histogram of total modified lines (additions + deletions) "
            "from a PR CSV file."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="inputs/github_pull_requests_optimization.csv",
        help=(
            "Input CSV with PRs (default: inputs/github_pull_requests_optimization.csv)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/pr_lines_histogram.png",
        help="Output image path (default: outputs/pr_lines_histogram.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30)",
    )
    return parser.parse_args()


def read_modified_lines(input_csv: str) -> List[int]:
    """Read additions and deletions from CSV and return total modified lines."""
    values: List[int] = []
    with open(input_csv, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if "additions" not in (reader.fieldnames or []):
            raise ValueError("Input CSV missing required column: additions")
        if "deletions" not in (reader.fieldnames or []):
            raise ValueError("Input CSV missing required column: deletions")

        for row in reader:
            additions_raw = (row.get("additions") or "").strip()
            deletions_raw = (row.get("deletions") or "").strip()
            try:
                additions = int(additions_raw)
                deletions = int(deletions_raw)
            except ValueError:
                continue
            total = max(additions + deletions, 0)
            values.append(total)

    return values


def main() -> int:
    args = parse_args()

    try:
        values = read_modified_lines(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        return 1
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if not values:
        print("Error: No valid additions/deletions data found.")
        return 1

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=args.bins, color="#2E5EAA", edgecolor="#1B1B1B")
    plt.title("Histogram of Modified Lines per PR")
    plt.xlabel("Modified Lines (additions + deletions)")
    plt.ylabel("Number of PRs")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

    print(f"Saved histogram to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
