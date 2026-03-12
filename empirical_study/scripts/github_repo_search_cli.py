"""CLI to search GitHub code with multiple queries and export unique repositories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import requests
from dotenv import load_dotenv

GITHUB_CODE_SEARCH_URL = "https://api.github.com/search/code"
PER_PAGE = 100
MAX_RESULTS_PER_QUERY = 1000
MAX_PAGES_PER_QUERY = MAX_RESULTS_PER_QUERY // PER_PAGE
SLEEP_SECONDS = 1
REQUEST_TIMEOUT_SECONDS = 30
CHECKPOINT_SUFFIX = ".checkpoint.json"
SIZE_SLICE_START = 0
SIZE_SLICE_STEP = 5000
MAX_SIZE_SLICES = 10000
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
            "Read multiple code-search queries from a CSV file, search GitHub code, "
            "and export unique repositories to another CSV file."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="inputs/input_queries.csv",
        help="Input CSV file containing queries (default: inputs/input_queries.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/github_repositories.csv",
        help="Output CSV file for unique repository results (default: outputs/github_repositories.csv)",
    )
    return parser.parse_args()


def read_queries(input_csv: str) -> list[str]:
    """Read queries from a CSV file without header using the first column only."""
    queries: list[str] = []

    with open(input_csv, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value:
                queries.append(value)

    return list(dict.fromkeys(queries))


def checkpoint_path_for_output(output_csv: str) -> str:
    """Return checkpoint file path derived from output CSV path."""
    return f"{output_csv}{CHECKPOINT_SUFFIX}"


def save_checkpoint(
    checkpoint_path: str,
    input_csv: str,
    output_csv: str,
    completed_queries: set[str],
    repos_by_full_name: dict[str, dict[str, object]],
    in_progress: dict[str, object] | None = None,
) -> None:
    """Persist resumable progress to disk."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    serialized_repos: dict[str, dict[str, object]] = {}
    for full_name, row in repos_by_full_name.items():
        serialized_row = dict(row)
        serialized_row["_matched_queries_set"] = sorted(
            serialized_row.get("_matched_queries_set", set())
        )
        serialized_repos[full_name] = serialized_row

    payload = {
        "version": 2,
        "input_csv": input_csv,
        "output_csv": output_csv,
        "completed_queries": sorted(completed_queries),
        "repos_by_full_name": serialized_repos,
        "in_progress": in_progress,
        "saved_at_epoch": int(time.time()),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, separators=(",", ":"))
    print(
        f"[CHECKPOINT] saved file={checkpoint_path} "
        f"completed_queries={len(completed_queries)} repos={len(repos_by_full_name)}"
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

    completed_queries = set(payload.get("completed_queries", []))
    raw_repos = payload.get("repos_by_full_name", {})
    repos_by_full_name: dict[str, dict[str, object]] = {}
    for full_name, row_obj in raw_repos.items():
        row = dict(row_obj)
        row["_matched_queries_set"] = set(row.get("_matched_queries_set", []))
        row["_query_match_counts"] = dict(row.get("_query_match_counts", {}))
        repos_by_full_name[full_name] = row
    in_progress = payload.get("in_progress")

    print(
        f"[CHECKPOINT] loaded file={checkpoint_path} "
        f"completed_queries={len(completed_queries)} repos={len(repos_by_full_name)}"
    )
    return completed_queries, repos_by_full_name, in_progress


def materialize_output_rows(
    repos_by_full_name: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Convert internal aggregate state into final CSV rows."""
    output_rows: list[dict[str, object]] = []
    for row in repos_by_full_name.values():
        matched_queries_sorted = sorted(row["_matched_queries_set"])
        query_breakdown_sorted = {
            key: row["_query_match_counts"][key]
            for key in sorted(row["_query_match_counts"].keys())
        }

        row["matched_queries_count"] = len(matched_queries_sorted)
        row["matched_queries"] = " | ".join(matched_queries_sorted)
        row["query_match_breakdown"] = json.dumps(
            query_breakdown_sorted,
            ensure_ascii=True,
            separators=(",", ":"),
        )

        row.pop("_matched_queries_set", None)
        row.pop("_query_match_counts", None)
        output_rows.append(row)

    output_rows.sort(
        key=lambda r: (
            r.get("matched_queries_count", 0),
            r.get("matched_code_results_count", 0),
            r.get("stargazers_count") or 0,
        ),
        reverse=True,
    )
    return output_rows


def github_api_request(
    url: str, token: str, params: dict[str, str | int], query: str, page: int
) -> dict[str, object]:
    """Perform a GitHub API GET request and return JSON payload."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "github-repo-search-cli",
    }
    print(
        f"[REQUEST] endpoint=/search/code query={query!r} page={page} "
        f"params={params}"
    )
    while True:
        try:
            response = requests.get(
                url=url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            sleep_non_api_error(
                context=f"query={query!r} page={page} network_error",
                exc=exc,
            )
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

        if maybe_wait_for_secondary_rate_limit(
            response=response,
            context=f"query={query!r} page={page}",
        ):
            continue

        if response.status_code == 403 and maybe_wait_for_rate_limit(
            rate_limit_remaining=remaining,
            rate_limit_reset=reset,
            context=f"query={query!r} page={page}",
        ):
            continue

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"GitHub API error {response.status_code}: {response.text}"
            ) from exc

        maybe_wait_for_rate_limit(
            rate_limit_remaining=remaining,
            rate_limit_reset=reset,
            context=f"post-success query={query!r} page={page}",
        )
        reset_secondary_backoff()
        try:
            payload = response.json()
        except ValueError as exc:
            sleep_non_api_error(
                context=f"query={query!r} page={page} json_error",
                exc=exc,
            )
            continue
        return payload


def build_size_slice_query(base_query: str, start_size: int, end_size: int) -> str:
    """Build one size-sliced query for GitHub code search."""
    return f"{base_query} size:{start_size}..{end_size}".strip()


def count_query_results(
    repos_by_full_name: dict[str, dict[str, object]],
    query: str,
) -> tuple[int, int]:
    """Compute total match count and unique repo count for a query."""
    total_matches = 0
    unique_repos = 0
    for row in repos_by_full_name.values():
        per_query = row.get("_query_match_counts", {})
        if query in per_query:
            unique_repos += 1
            total_matches += int(per_query[query])
    return total_matches, unique_repos


def init_repo_row(repo: dict[str, object]) -> dict[str, object]:
    """Create the base output row for a repository from GitHub payload fields."""
    owner = repo.get("owner") or {}
    license_obj = repo.get("license") or {}
    return {
        "repo_id": repo.get("id"),
        "full_name": repo.get("full_name"),
        "name": repo.get("name"),
        "repo_html_url": repo.get("html_url"),
        "description": repo.get("description"),
        "language": repo.get("language"),
        "stargazers_count": repo.get("stargazers_count"),
        "forks_count": repo.get("forks_count"),
        "open_issues_count": repo.get("open_issues_count"),
        "watchers_count": repo.get("watchers_count"),
        "default_branch": repo.get("default_branch"),
        "owner_login": owner.get("login"),
        "owner_type": owner.get("type"),
        "license": license_obj.get("spdx_id") or license_obj.get("name"),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "is_private": repo.get("private"),
        "is_fork": repo.get("fork"),
        "archived": repo.get("archived"),
        "matched_queries_count": 0,
        "matched_code_results_count": 0,
        "matched_queries": "",
        "query_match_breakdown": "",
        "_matched_queries_set": set(),
        "_query_match_counts": {},
    }


def apply_code_item_to_repo(
    repository: dict[str, object],
    query: str,
    repos_by_full_name: dict[str, dict[str, object]],
) -> None:
    """Merge one code search item into repository aggregates."""
    full_name = repository.get("full_name")
    if not full_name:
        return

    row = repos_by_full_name.get(full_name)
    if row is None:
        row = init_repo_row(repository)
        repos_by_full_name[full_name] = row

    row["matched_code_results_count"] += 1
    row["_matched_queries_set"].add(query)
    match_counts = row["_query_match_counts"]
    match_counts[query] = match_counts.get(query, 0) + 1


def process_query_with_slices(
    base_query: str,
    token: str,
    repos_by_full_name: dict[str, dict[str, object]],
    checkpoint_path: str,
    input_csv: str,
    output_csv: str,
    completed_queries: set[str],
    resume_state: dict[str, object] | None,
) -> None:
    """Fetch a base query across size slices, checkpointing after each request."""
    start_slice_idx = 0
    start_slice_start = SIZE_SLICE_START
    start_page = 1
    start_emitted = 0

    if resume_state and resume_state.get("query") == base_query:
        start_slice_idx = int(resume_state.get("slice_idx", 0))
        start_slice_start = int(resume_state.get("slice_start", SIZE_SLICE_START))
        start_page = int(resume_state.get("page", 1))
        start_emitted = int(resume_state.get("emitted", 0))

    slice_start = start_slice_start
    for slice_idx in range(start_slice_idx, MAX_SIZE_SLICES):
        slice_end = slice_start + SIZE_SLICE_STEP - 1
        sliced_query = build_size_slice_query(base_query, slice_start, slice_end)

        page = start_page if slice_idx == start_slice_idx else 1
        emitted = start_emitted if slice_idx == start_slice_idx else 0
        first_page_total_count = 0
        slice_result_count = 0
        slice_unique_repos: set[str] = set()

        while page <= MAX_PAGES_PER_QUERY and emitted < MAX_RESULTS_PER_QUERY:
            params = {
                "q": sliced_query,
                "per_page": PER_PAGE,
                "page": page,
            }
            try:
                payload = github_api_request(
                    GITHUB_CODE_SEARCH_URL,
                    token,
                    params,
                    query=sliced_query,
                    page=page,
                )
            except RuntimeError as exc:
                in_progress = {
                    "query": base_query,
                    "slice_idx": slice_idx,
                    "slice_start": slice_start,
                    "page": page,
                    "emitted": emitted,
                }
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=input_csv,
                    output_csv=output_csv,
                    completed_queries=completed_queries,
                    repos_by_full_name=repos_by_full_name,
                    in_progress=in_progress,
                )
                raise exc
            except Exception as exc:
                in_progress = {
                    "query": base_query,
                    "slice_idx": slice_idx,
                    "slice_start": slice_start,
                    "page": page,
                    "emitted": emitted,
                }
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=input_csv,
                    output_csv=output_csv,
                    completed_queries=completed_queries,
                    repos_by_full_name=repos_by_full_name,
                    in_progress=in_progress,
                )
                sleep_non_api_error(
                    context=f"query={sliced_query!r} page={page}",
                    exc=exc,
                )
                continue

            if page == 1:
                first_page_total_count = int(payload.get("total_count", 0))

            items = payload.get("items", [])
            if not items:
                in_progress = {
                    "query": base_query,
                    "slice_idx": slice_idx,
                    "slice_start": slice_start,
                    "page": page + 1,
                    "emitted": emitted,
                }
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=input_csv,
                    output_csv=output_csv,
                    completed_queries=completed_queries,
                    repos_by_full_name=repos_by_full_name,
                    in_progress=in_progress,
                )
                break

            for item in items:
                if emitted >= MAX_RESULTS_PER_QUERY:
                    break
                emitted += 1
                repository = item.get("repository") or {}
                full_name = repository.get("full_name")
                if not full_name:
                    continue
                slice_result_count += 1
                slice_unique_repos.add(full_name)
                apply_code_item_to_repo(repository, base_query, repos_by_full_name)

            in_progress = {
                "query": base_query,
                "slice_idx": slice_idx,
                "slice_start": slice_start,
                "page": page + 1,
                "emitted": emitted,
            }
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                input_csv=input_csv,
                output_csv=output_csv,
                completed_queries=completed_queries,
                repos_by_full_name=repos_by_full_name,
                in_progress=in_progress,
            )

            if len(items) < PER_PAGE:
                break

            page += 1
            time.sleep(max(SLEEP_SECONDS, 0))

        if emitted >= MAX_RESULTS_PER_QUERY:
            print(
                f"[QUERY_LIMIT] query={sliced_query!r} reached limit="
                f"{MAX_RESULTS_PER_QUERY} results, stopping pagination for this search "
                "query."
            )

        print(
            f"[SLICE] base_query={base_query!r} index={slice_idx} "
            f"size={slice_start}..{slice_end} total_count={first_page_total_count} "
            f"fetched_items={slice_result_count} unique_repos={len(slice_unique_repos)}"
        )
        if first_page_total_count == 0:
            print(
                f"[SLICE_STOP] base_query={base_query!r} reached zero-result slice "
                f"at size={slice_start}..{slice_end}"
            )
            break

        slice_start = slice_end + 1
        start_page = 1
        start_emitted = 0
        in_progress = {
            "query": base_query,
            "slice_idx": slice_idx + 1,
            "slice_start": slice_start,
            "page": 1,
            "emitted": 0,
        }
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            input_csv=input_csv,
            output_csv=output_csv,
            completed_queries=completed_queries,
            repos_by_full_name=repos_by_full_name,
            in_progress=in_progress,
        )
    else:
        print(
            f"[SLICE_STOP] base_query={base_query!r} reached max slices "
            f"limit={MAX_SIZE_SLICES}."
        )


def aggregate_unique_repositories(
    queries: list[str],
    token: str,
    input_csv: str,
    output_csv: str,
) -> tuple[list[dict[str, object]], bool]:
    """Aggregate code-search matches into unique repositories across all queries."""
    checkpoint_path = checkpoint_path_for_output(output_csv)
    completed_queries, repos_by_full_name, in_progress = load_checkpoint(
        checkpoint_path=checkpoint_path,
        input_csv=input_csv,
        output_csv=output_csv,
    )
    interrupted = False

    for query in queries:
        if query in completed_queries:
            print(f"[RESUME] skipping already completed query: {query}")
            continue

        print(f"Searching code query: {query}")
        try:
            resume_state = None
            if (
                in_progress
                and in_progress.get("query") == query
                and query not in completed_queries
            ):
                resume_state = in_progress
                print(
                    f"[RESUME] continuing query={query!r} "
                    f"slice_idx={resume_state.get('slice_idx')} "
                    f"page={resume_state.get('page')}"
                )

            process_query_with_slices(
                base_query=query,
                token=token,
                repos_by_full_name=repos_by_full_name,
                checkpoint_path=checkpoint_path,
                input_csv=input_csv,
                output_csv=output_csv,
                completed_queries=completed_queries,
                resume_state=resume_state,
            )
        except KeyboardInterrupt:
            interrupted = True
            print(
                f"[INTERRUPT] detected during query={query!r}; "
                "current query progress not committed. Saving checkpoint."
            )
            if not os.path.exists(checkpoint_path):
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_csv=input_csv,
                    output_csv=output_csv,
                    completed_queries=completed_queries,
                    repos_by_full_name=repos_by_full_name,
                    in_progress={"query": query},
                )
            break

        query_result_count, query_unique_repo_count = count_query_results(
            repos_by_full_name=repos_by_full_name,
            query=query,
        )
        print(
            "  -> code matches: "
            f"{query_result_count}, unique repos in this query: "
            f"{query_unique_repo_count}"
        )
        completed_queries.add(query)
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            input_csv=input_csv,
            output_csv=output_csv,
            completed_queries=completed_queries,
            repos_by_full_name=repos_by_full_name,
            in_progress=None,
        )
        in_progress = None

    output_rows = materialize_output_rows(repos_by_full_name)
    if not interrupted and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[CHECKPOINT] removed after successful completion: {checkpoint_path}")
    return output_rows, interrupted


def write_results(output_csv: str, rows: list[dict[str, object]]) -> None:
    """Write aggregated repository rows to the output CSV file."""
    headers = [
        "repo_id",
        "full_name",
        "name",
        "repo_html_url",
        "description",
        "language",
        "stargazers_count",
        "forks_count",
        "open_issues_count",
        "watchers_count",
        "default_branch",
        "owner_login",
        "owner_type",
        "license",
        "created_at",
        "updated_at",
        "pushed_at",
        "is_private",
        "is_fork",
        "archived",
        "matched_queries_count",
        "matched_code_results_count",
        "matched_queries",
        "query_match_breakdown",
    ]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
        queries = read_queries(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not queries:
        print("Error: No queries found in input file.", file=sys.stderr)
        return 1

    all_rows, interrupted = aggregate_unique_repositories(
        queries=queries,
        token=token,
        input_csv=args.input,
        output_csv=args.output,
    )

    write_results(args.output, all_rows)
    if interrupted:
        print(
            f"[PARTIAL_OUTPUT] saved {len(all_rows)} unique repositories to: {args.output}"
        )
        print(
            "Run was interrupted. Resume by executing the same command again."
        )
        return 130

    print(f"Saved {len(all_rows)} unique repositories to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
