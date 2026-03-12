"""CLI to filter PRs by optimization keywords using GitHub PR details."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

GITHUB_PR_DETAIL_URL = "https://api.github.com/repos/{full_name}/pulls/{number}"
PER_PAGE = 100
SLEEP_SECONDS = 1
REQUEST_TIMEOUT_SECONDS = 30
CHECKPOINT_SUFFIX = ".checkpoint.json"
SECONDARY_BACKOFF_BASE_SECONDS = 10
SECONDARY_BACKOFF_CAP_SECONDS = 600
_secondary_backoff_seconds = SECONDARY_BACKOFF_BASE_SECONDS
NON_API_RETRY_SLEEP_SECONDS = 60

DEFAULT_KEYWORDS = [
    "optimiz",
    "optimization",
    "optimize",
    "performance",
    "speed",
    "latency",
    "throughput",
    "efficient",
    "efficiency",
    "memory",
    "cache",
    "vectoriz",
    "profil",
    "bottleneck",
]


def maybe_wait_for_rate_limit(
    rate_limit_remaining: str | None,
    rate_limit_reset: str | None,
    context: str,
) -> bool:
    """Sleep until reset time when rate limit is exhausted; return True if slept."""
    if rate_limit_remaining != "0":
        return False
    if not rate_limit_reset:
        return False
    try:
        reset_epoch = int(rate_limit_reset)
    except ValueError:
        return False

    now_epoch = int(time.time())
    wait_seconds = max(reset_epoch - now_epoch + 1, 1)
    print(
        f"[RATE_LIMIT] {context} remaining=0, sleeping {wait_seconds}s "
        f"until reset_epoch={reset_epoch}"
    )
    time.sleep(wait_seconds)
    return True


def maybe_wait_for_secondary_rate_limit(
    response: requests.Response,
    context: str,
) -> bool:
    """Sleep when GitHub secondary rate limit is hit; return True if slept."""
    if response.status_code != 429 and response.status_code != 403:
        return False

    body = (response.text or "").lower()
    if response.status_code == 403 and "secondary rate limit" not in body:
        return False

    global _secondary_backoff_seconds
    retry_after = response.headers.get("Retry-After")
    reset = response.headers.get("X-RateLimit-Reset")
    wait_seconds = _secondary_backoff_seconds
    if retry_after:
        try:
            wait_seconds = max(int(retry_after), wait_seconds, 1)
        except ValueError:
            pass
    elif reset:
        try:
            reset_epoch = int(reset)
            now_epoch = int(time.time())
            wait_seconds = max(reset_epoch - now_epoch + 1, wait_seconds, 1)
        except ValueError:
            pass

    next_backoff = min(SECONDARY_BACKOFF_CAP_SECONDS, max(wait_seconds * 2, 1))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(
        f"[SECONDARY_RATE_LIMIT] {timestamp} {context} sleeping {wait_seconds}s "
        f"status={response.status_code} next_backoff={next_backoff}s"
    )
    time.sleep(wait_seconds)
    _secondary_backoff_seconds = next_backoff
    return True


def reset_secondary_backoff() -> None:
    """Reset exponential backoff after a successful request."""
    global _secondary_backoff_seconds
    _secondary_backoff_seconds = SECONDARY_BACKOFF_BASE_SECONDS


def sleep_non_api_error(context: str, exc: Exception) -> None:
    """Sleep after non-API error and retry indefinitely."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(
        f"[NON_API_ERROR] {timestamp} {context} error={exc!r} "
        f"sleeping={NON_API_RETRY_SLEEP_SECONDS}s"
    )
    time.sleep(NON_API_RETRY_SLEEP_SECONDS)


def normalize_text(value: str) -> str:
    """Normalize text for keyword matching (lower + strip accents)."""
    lower = value.lower()
    return (
        lower.replace("á", "a")
        .replace("à", "a")
        .replace("ä", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ë", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("ï", "i")
        .replace("î", "i")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ö", "o")
        .replace("ô", "o")
        .replace("ú", "u")
        .replace("ù", "u")
        .replace("ü", "u")
        .replace("û", "u")
        .replace("ñ", "n")
    )


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read PRs from a CSV (output of github_repo_prs_cli.py), fetch PR details, "
            "and export PRs that mention optimization keywords."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="inputs/github_pull_requests.csv",
        help=(
            "Input CSV with PRs (default: inputs/github_pull_requests.csv)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/github_pull_requests_optimization.csv",
        help=(
            "Output CSV for optimization-related PRs "
            "(default: outputs/github_pull_requests_optimization.csv)"
        ),
    )
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help=(
            "Comma-separated list of keyword fragments to match "
            f"(default: {','.join(DEFAULT_KEYWORDS)})"
        ),
    )
    return parser.parse_args()


def read_prs(input_csv: str) -> list[dict[str, str]]:
    """Read PR entries from input CSV."""
    prs: list[dict[str, str]] = []
    with open(input_csv, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if "repo_full_name" not in (reader.fieldnames or []):
            raise ValueError("Input CSV missing required column: repo_full_name")
        if "pr_number" not in (reader.fieldnames or []):
            raise ValueError("Input CSV missing required column: pr_number")
        for row in reader:
            repo_full_name = (row.get("repo_full_name") or "").strip()
            pr_number = (row.get("pr_number") or "").strip()
            if not repo_full_name or not pr_number:
                continue
            prs.append(
                {
                    "pr_id": (row.get("pr_id") or "").strip(),
                    "pr_number": pr_number,
                    "pr_html_url": (row.get("pr_html_url") or "").strip(),
                    "title": (row.get("title") or "").strip(),
                    "state": (row.get("state") or "").strip(),
                    "is_draft": (row.get("is_draft") or "").strip(),
                    "user_login": (row.get("user_login") or "").strip(),
                    "repo_full_name": repo_full_name,
                    "base_branch": (row.get("base_branch") or "").strip(),
                    "head_branch": (row.get("head_branch") or "").strip(),
                    "created_at": (row.get("created_at") or "").strip(),
                    "updated_at": (row.get("updated_at") or "").strip(),
                    "closed_at": (row.get("closed_at") or "").strip(),
                    "merged_at": (row.get("merged_at") or "").strip(),
                }
            )
    return prs


def checkpoint_path_for_output(output_csv: str) -> str:
    """Return checkpoint file path derived from output CSV path."""
    return f"{output_csv}{CHECKPOINT_SUFFIX}"


def save_checkpoint(
    checkpoint_path: str,
    input_csv: str,
    output_csv: str,
    completed_pr_ids: set[str],
    matched_prs_by_id: dict[str, dict[str, object]],
    in_progress: dict[str, object] | None = None,
) -> None:
    """Persist resumable progress to disk."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    payload = {
        "version": 1,
        "input_csv": input_csv,
        "output_csv": output_csv,
        "completed_pr_ids": sorted(completed_pr_ids),
        "matched_prs_by_id": matched_prs_by_id,
        "in_progress": in_progress,
        "saved_at_epoch": int(time.time()),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, separators=(",", ":"))
    print(
        f"[CHECKPOINT] saved file={checkpoint_path} "
        f"completed_prs={len(completed_pr_ids)} matched={len(matched_prs_by_id)}"
    )


def load_checkpoint(
    checkpoint_path: str,
    input_csv: str,
    output_csv: str,
) -> tuple[set[str], dict[str, dict[str, object]], dict[str, object] | None]:
    """Load checkpoint if available and compatible with current run."""
    if not os.path.exists(checkpoint_path):
        return set(), {}, None

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if payload.get("input_csv") != input_csv or payload.get("output_csv") != output_csv:
        print(
            "[CHECKPOINT] existing checkpoint ignored due to input/output mismatch: "
            f"{checkpoint_path}"
        )
        return set(), {}, None

    completed_pr_ids = set(payload.get("completed_pr_ids", []))
    matched_prs_by_id = payload.get("matched_prs_by_id", {})
    in_progress = payload.get("in_progress")

    print(
        f"[CHECKPOINT] loaded file={checkpoint_path} "
        f"completed_prs={len(completed_pr_ids)} matched={len(matched_prs_by_id)}"
    )
    return completed_pr_ids, matched_prs_by_id, in_progress


def github_api_request(
    url: str, token: str, params: dict[str, str | int], context: str
) -> requests.Response:
    """Perform a GitHub API GET request and return response."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "github-pr-optimization-filter-cli",
    }
    print(f"[REQUEST] {context} params={params}")
    while True:
        try:
            response = requests.get(
                url=url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            sleep_non_api_error(context=f"{context} network_error", exc=exc)
            continue

        rate_limit = response.headers.get("X-RateLimit-Limit")
        remaining = response.headers.get("X-RateLimit-Remaining")
        used = response.headers.get("X-RateLimit-Used")
        reset = response.headers.get("X-RateLimit-Reset")
        print(
            "[RESPONSE] "
            f"status={response.status_code} "
            f"rate_limit={rate_limit} "
            f"remaining={remaining} "
            f"used={used} "
            f"reset={reset}"
        )

        if maybe_wait_for_secondary_rate_limit(response=response, context=context):
            continue

        if response.status_code == 403 and maybe_wait_for_rate_limit(
            rate_limit_remaining=remaining,
            rate_limit_reset=reset,
            context=context,
        ):
            continue

        return response


def keyword_matches(text: str, keywords: list[str]) -> list[str]:
    """Return matching keywords present in text."""
    haystack = normalize_text(text)
    matches = [kw for kw in keywords if kw and kw in haystack]
    return sorted(set(matches))


def pr_row_from_payload(
    pr: dict[str, object],
    base_row: dict[str, str],
    matched_keywords: list[str],
) -> dict[str, object]:
    """Build output row for a PR from GitHub payload."""
    return {
        **base_row,
        "matched_keywords": " | ".join(matched_keywords),
        "body_snippet": (pr.get("body") or "").strip()[:400],
        "comments": pr.get("comments"),
        "review_comments": pr.get("review_comments"),
        "commits": pr.get("commits"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changed_files"),
        "mergeable_state": pr.get("mergeable_state"),
    }


def write_results(output_csv: str, matched_prs_by_id: dict[str, dict[str, object]]) -> None:
    """Write matched PRs to output CSV."""
    headers = [
        "pr_id",
        "pr_number",
        "pr_html_url",
        "title",
        "state",
        "is_draft",
        "user_login",
        "repo_full_name",
        "base_branch",
        "head_branch",
        "created_at",
        "updated_at",
        "closed_at",
        "merged_at",
        "matched_keywords",
        "body_snippet",
        "comments",
        "review_comments",
        "commits",
        "additions",
        "deletions",
        "changed_files",
        "mergeable_state",
    ]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = list(matched_prs_by_id.values())
    rows.sort(key=lambda r: (r.get("repo_full_name") or "", r.get("pr_number") or 0))

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Run the CLI workflow end-to-end and return a process exit code."""
    args = parse_args()
    load_dotenv()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print(
            "Error: Missing token. Set GITHUB_TOKEN in .env or environment.",
            file=sys.stderr,
        )
        return 1
    print(
        f"[AUTH] token_detected=yes token_length={len(token)} "
        f"token_prefix={token[:4] if len(token) >= 4 else token!r}"
    )

    keywords = [kw.strip().lower() for kw in args.keywords.split(",") if kw.strip()]
    if not keywords:
        print("Error: No keywords provided.", file=sys.stderr)
        return 1

    try:
        prs = read_prs(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not prs:
        print("Error: No PRs found in input file.", file=sys.stderr)
        return 1

    checkpoint_path = checkpoint_path_for_output(args.output)
    completed_pr_ids, matched_prs_by_id, in_progress = load_checkpoint(
        checkpoint_path=checkpoint_path,
        input_csv=args.input,
        output_csv=args.output,
    )

    start_index = 0
    if in_progress and "next_index" in in_progress:
        try:
            start_index = int(in_progress["next_index"])
        except (TypeError, ValueError):
            start_index = 0

    interrupted = False
    for idx in range(start_index, len(prs)):
        base_row = prs[idx]
        pr_id = base_row.get("pr_id") or f"{base_row['repo_full_name']}#{base_row['pr_number']}"
        if pr_id in completed_pr_ids:
            continue

        repo_full_name = base_row["repo_full_name"]
        pr_number = base_row["pr_number"]
        context = f"repo={repo_full_name} pr={pr_number}"

        try:
            url = GITHUB_PR_DETAIL_URL.format(
                full_name=repo_full_name, number=pr_number
            )
            response = github_api_request(url, token, params={}, context=context)

            if response.status_code == 404:
                print(f"[SKIP] PR not found or inaccessible: {context}")
                completed_pr_ids.add(pr_id)
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=args.input,
                    output_csv=args.output,
                    completed_pr_ids=completed_pr_ids,
                    matched_prs_by_id=matched_prs_by_id,
                    in_progress={"next_index": idx + 1},
                )
                continue

            response.raise_for_status()
            reset_secondary_backoff()

            try:
                pr = response.json()
            except ValueError as exc:
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=args.input,
                    output_csv=args.output,
                    completed_pr_ids=completed_pr_ids,
                    matched_prs_by_id=matched_prs_by_id,
                    in_progress={"next_index": idx},
                )
                sleep_non_api_error(context=f"{context} json_error", exc=exc)
                continue
            title = str(pr.get("title") or "")
            body = str(pr.get("body") or "")
            matches = keyword_matches(f"{title}\n{body}", keywords)

            if matches:
                matched_prs_by_id[str(pr_id)] = pr_row_from_payload(
                    pr=pr, base_row=base_row, matched_keywords=matches
                )
                print(f"[MATCH] {context} keywords={matches}")

            completed_pr_ids.add(pr_id)
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                input_csv=args.input,
                output_csv=args.output,
                completed_pr_ids=completed_pr_ids,
                matched_prs_by_id=matched_prs_by_id,
                in_progress={"next_index": idx + 1},
            )
            time.sleep(max(SLEEP_SECONDS, 0))
        except KeyboardInterrupt:
            interrupted = True
            print(f"[INTERRUPT] detected during {context}; saving checkpoint.")
            if not os.path.exists(checkpoint_path):
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=args.input,
                    output_csv=args.output,
                    completed_pr_ids=completed_pr_ids,
                    matched_prs_by_id=matched_prs_by_id,
                    in_progress={"next_index": idx},
                )
            break
        except requests.exceptions.HTTPError as exc:
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                input_csv=args.input,
                output_csv=args.output,
                completed_pr_ids=completed_pr_ids,
                matched_prs_by_id=matched_prs_by_id,
                in_progress={"next_index": idx},
            )
            raise RuntimeError(f"GitHub API error: {exc}") from exc
        except Exception as exc:
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                input_csv=args.input,
                output_csv=args.output,
                completed_pr_ids=completed_pr_ids,
                matched_prs_by_id=matched_prs_by_id,
                in_progress={"next_index": idx},
            )
            sleep_non_api_error(context=context, exc=exc)
            continue

    write_results(args.output, matched_prs_by_id)
    if not interrupted and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[CHECKPOINT] removed after successful completion: {checkpoint_path}")

    if interrupted:
        print(
            f"[PARTIAL_OUTPUT] saved {len(matched_prs_by_id)} matched PRs to: "
            f"{args.output}"
        )
        print("Run was interrupted. Resume by executing the same command again.")
        return 130

    print(f"Saved {len(matched_prs_by_id)} matched PRs to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
