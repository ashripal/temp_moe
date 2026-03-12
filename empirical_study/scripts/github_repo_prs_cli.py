"""CLI to list pull requests for repositories from a CSV and export unique PRs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

GITHUB_REPO_PRS_URL = "https://api.github.com/repos/{full_name}/pulls"
PER_PAGE = 100
SLEEP_SECONDS = 0.1
REQUEST_TIMEOUT_SECONDS = 30
CHECKPOINT_SUFFIX = ".checkpoint.json"
SECONDARY_BACKOFF_BASE_SECONDS = 10
SECONDARY_BACKOFF_CAP_SECONDS = 600
_secondary_backoff_seconds = SECONDARY_BACKOFF_BASE_SECONDS
NON_API_RETRY_SLEEP_SECONDS = 60


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


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read repositories from a CSV (output of github_repo_search_cli.py), "
            "list pull requests per repository, and export unique PRs to another CSV."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="outputs/github_repositories.csv",
        help=(
            "Input CSV with repositories (default: outputs/github_repositories.csv)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/github_pull_requests.csv",
        help=(
            "Output CSV for unique pull requests (default: outputs/github_pull_requests.csv)"
        ),
    )
    return parser.parse_args()


def read_repos(input_csv: str) -> list[str]:
    """Read repository full_name entries from an input CSV."""
    repos: list[str] = []
    with open(input_csv, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if "full_name" not in (reader.fieldnames or []):
            raise ValueError("Input CSV missing required column: full_name")
        for row in reader:
            full_name = (row.get("full_name") or "").strip()
            if full_name:
                repos.append(full_name)
    return list(dict.fromkeys(repos))


def checkpoint_path_for_output(output_csv: str) -> str:
    """Return checkpoint file path derived from output CSV path."""
    return f"{output_csv}{CHECKPOINT_SUFFIX}"


def save_checkpoint(
    checkpoint_path: str,
    input_csv: str,
    output_csv: str,
    completed_repos: set[str],
    prs_by_id: dict[str, dict[str, object]],
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
        "completed_repos": sorted(completed_repos),
        "prs_by_id": prs_by_id,
        "in_progress": in_progress,
        "saved_at_epoch": int(time.time()),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, separators=(",", ":"))
    print(
        f"[CHECKPOINT] saved file={checkpoint_path} "
        f"completed_repos={len(completed_repos)} prs={len(prs_by_id)}"
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

    completed_repos = set(payload.get("completed_repos", []))
    prs_by_id = payload.get("prs_by_id", {})
    in_progress = payload.get("in_progress")

    print(
        f"[CHECKPOINT] loaded file={checkpoint_path} "
        f"completed_repos={len(completed_repos)} prs={len(prs_by_id)}"
    )
    return completed_repos, prs_by_id, in_progress


def github_api_request(
    url: str, token: str, params: dict[str, str | int], context: str
) -> requests.Response:
    """Perform a GitHub API GET request and return response."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "github-repo-prs-cli",
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


def pr_row_from_payload(pr: dict[str, object], repo_full_name: str) -> dict[str, object]:
    """Build output row for a PR from GitHub payload."""
    user = pr.get("user") or {}
    base = pr.get("base") or {}
    head = pr.get("head") or {}
    return {
        "pr_id": pr.get("id"),
        "pr_number": pr.get("number"),
        "pr_html_url": pr.get("html_url"),
        "title": pr.get("title"),
        "state": pr.get("state"),
        "is_draft": pr.get("draft"),
        "user_login": user.get("login"),
        "repo_full_name": repo_full_name,
        "base_branch": base.get("ref"),
        "head_branch": head.get("ref"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "closed_at": pr.get("closed_at"),
        "merged_at": pr.get("merged_at"),
    }


def fetch_repo_prs(
    repo_full_name: str,
    token: str,
    prs_by_id: dict[str, dict[str, object]],
    checkpoint_path: str,
    input_csv: str,
    output_csv: str,
    completed_repos: set[str],
    resume_state: dict[str, object] | None,
) -> None:
    """Fetch PRs for one repository with pagination and checkpointing."""
    page = 1
    if resume_state and resume_state.get("repo_full_name") == repo_full_name:
        page = int(resume_state.get("page", 1))

    while True:
        url = GITHUB_REPO_PRS_URL.format(full_name=repo_full_name)
        params = {"state": "all", "per_page": PER_PAGE, "page": page}
        context = f"repo={repo_full_name} page={page}"
        response = github_api_request(url, token, params, context=context)

        if response.status_code == 404:
            print(f"[SKIP] repo not found or inaccessible: {repo_full_name}")
            break

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"GitHub API error {response.status_code}: {response.text}"
            ) from exc
        reset_secondary_backoff()

        try:
            payload = response.json()
        except ValueError as exc:
            sleep_non_api_error(context=f"{context} json_error", exc=exc)
            continue
        if not payload:
            in_progress = {
                "repo_full_name": repo_full_name,
                "page": page + 1,
            }
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                input_csv=input_csv,
                output_csv=output_csv,
                completed_repos=completed_repos,
                prs_by_id=prs_by_id,
                in_progress=in_progress,
            )
            break

        for pr in payload:
            pr_id = pr.get("id")
            if pr_id is None:
                continue
            key = str(pr_id)
            if key not in prs_by_id:
                prs_by_id[key] = pr_row_from_payload(pr, repo_full_name)

        in_progress = {"repo_full_name": repo_full_name, "page": page + 1}
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            input_csv=input_csv,
            output_csv=output_csv,
            completed_repos=completed_repos,
            prs_by_id=prs_by_id,
            in_progress=in_progress,
        )

        if len(payload) < PER_PAGE:
            break

        page += 1
        time.sleep(max(SLEEP_SECONDS, 0))


def write_results(output_csv: str, prs_by_id: dict[str, dict[str, object]]) -> None:
    """Write unique pull requests to output CSV."""
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
    ]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = list(prs_by_id.values())
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

    try:
        repos = read_repos(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not repos:
        print("Error: No repositories found in input file.", file=sys.stderr)
        return 1

    checkpoint_path = checkpoint_path_for_output(args.output)
    completed_repos, prs_by_id, in_progress = load_checkpoint(
        checkpoint_path=checkpoint_path,
        input_csv=args.input,
        output_csv=args.output,
    )

    interrupted = False
    for repo_full_name in repos:
        if repo_full_name in completed_repos:
            print(f"[RESUME] skipping already completed repo: {repo_full_name}")
            continue

        print(f"Listing PRs for repo: {repo_full_name}")
        try:
            resume_state = None
            if (
                in_progress
                and in_progress.get("repo_full_name") == repo_full_name
                and repo_full_name not in completed_repos
            ):
                resume_state = in_progress
                print(
                    f"[RESUME] continuing repo={repo_full_name!r} "
                    f"page={resume_state.get('page')}"
                )

            fetch_repo_prs(
                repo_full_name=repo_full_name,
                token=token,
                prs_by_id=prs_by_id,
                checkpoint_path=checkpoint_path,
                input_csv=args.input,
                output_csv=args.output,
                completed_repos=completed_repos,
                resume_state=resume_state,
            )
        except KeyboardInterrupt:
            interrupted = True
            print(
                f"[INTERRUPT] detected during repo={repo_full_name!r}; "
                "current repo progress not committed."
            )
            if not os.path.exists(checkpoint_path):
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=args.input,
                    output_csv=args.output,
                    completed_repos=completed_repos,
                    prs_by_id=prs_by_id,
                    in_progress={"repo_full_name": repo_full_name},
                )
            break
        except Exception as exc:
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                input_csv=args.input,
                output_csv=args.output,
                completed_repos=completed_repos,
                prs_by_id=prs_by_id,
                in_progress={"repo_full_name": repo_full_name},
            )
            sleep_non_api_error(context=f"repo={repo_full_name}", exc=exc)
            continue

        completed_repos.add(repo_full_name)
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            input_csv=args.input,
            output_csv=args.output,
            completed_repos=completed_repos,
            prs_by_id=prs_by_id,
            in_progress=None,
        )
        in_progress = None

    write_results(args.output, prs_by_id)
    if not interrupted and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[CHECKPOINT] removed after successful completion: {checkpoint_path}")

    if interrupted:
        print(
            f"[PARTIAL_OUTPUT] saved {len(prs_by_id)} unique pull requests to: "
            f"{args.output}"
        )
        print("Run was interrupted. Resume by executing the same command again.")
        return 130

    print(f"Saved {len(prs_by_id)} unique pull requests to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
