# GitHub Code Search to Unique Repositories CLI

This project provides a Python CLI that reads a list of GitHub **code search queries** from an input CSV file, fetches code matches for each query, and writes a deduplicated list of repositories to an output CSV file.

## Features

- Reads multiple code queries from a CSV file.
- Uses a GitHub token loaded from `.env`.
- Uses `requests` for GitHub API calls.
- Uses `argparse` for CLI configuration.
- Lets you configure input and output file names.
- Searches GitHub **code** (`/search/code`) instead of repository search.
- Exports **unique repositories** that matched one or more queries.
- Includes per-repository query coverage metrics.
- Saves checkpoint progress and auto-resumes on next run after interruption.
- Uses automatic `size:` slicing per base query to go beyond a single 1000-result window.

## Files

- `scripts/github_repo_search_cli.py`: Main CLI script.
- `.env-template`: Template for environment variables.
- `.env`: Local token file (not for sharing).
- `inputs/input_queries.csv`: Dummy input example with sample queries.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.9+
- `python-dotenv` (listed in `requirements.txt`)
- `requests` (listed in `requirements.txt`)
- A GitHub Personal Access Token with access to the Search API.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create your token file:

```bash
cp .env-template .env
```

3. Edit `.env` and set your token:

```env
GITHUB_TOKEN=your_real_token_here
```

## Input Format

Example `inputs/input_queries.csv`:

```csv
MPI_Init language:C
MPI_Comm_rank language:C++
"#pragma omp parallel" language:C++
cudaMemcpy language:C++
```

The input file must not have a header. Each row is treated as one query (first column only).

## Usage

Default usage:

```bash
python3 scripts/github_repo_search_cli.py
```

Custom input/output files:

```bash
python3 scripts/github_repo_search_cli.py \
  --input inputs/input_queries.csv \
  --output outputs/github_repositories.csv
```

Show help:

```bash
python3 scripts/github_repo_search_cli.py --help
```

## Output

The output CSV contains one row per unique repository, with fields such as:

- `full_name`
- `repo_html_url`
- `description`
- `language`
- `stargazers_count`
- `matched_queries_count`
- `matched_code_results_count`
- `matched_queries`
- `query_match_breakdown`

## Resume Behavior

- The script writes a checkpoint file at:
  `outputs/github_repositories.csv.checkpoint.json` (based on your `--output` path).
- On restart with the same `--input` and `--output`, it automatically resumes and skips completed queries.
- If interrupted during a query (`Ctrl+C`), progress for that in-flight query is not committed, but completed queries are preserved.
- On full successful completion, the checkpoint file is removed automatically.

## Notes and Limits

- GitHub Search API limits search results to a maximum of 1000 items per query.
- The script uses `size` slices (`size:0..4999`, `5000..9999`, ...) and paginates each slice up to the API limit.
- Slicing stops for a base query when a slice returns `total_count=0` (or when safety cap is reached).
- The delay between paginated requests is fixed to `1` second.
- `matched_code_results_count` is the total number of code-result items seen for that repository across all queries.
- Keep your `.env` private and never commit real tokens.
