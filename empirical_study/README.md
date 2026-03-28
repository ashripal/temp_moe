# Empirical Study: GitHub HPC Optimization Analysis

This directory contains tools for empirical analysis of HPC optimization patterns from GitHub repositories, including code search, pull request analysis, and data visualization.

## Overview

The empirical study pipeline consists of three main stages:

1. **Code Search** - Find repositories with HPC-related code patterns
2. **PR Analysis** - Extract pull requests and optimization-related changes
3. **Visualization** - Generate histograms and analysis from PR data

## Directory Structure

```
empirical_study/
├── scripts/
│   ├── github_repo_search_cli.py         # Stage 1: Search for HPC code patterns
│   ├── github_repo_prs_cli.py            # Stage 2: Extract PRs from repositories
│   ├── github_pr_optimization_filter_cli.py  # Filter and analyze optimization PRs
│   └── pr_lines_histogram.py             # Stage 3: Visualize PR statistics
├── inputs/
│   ├── input_queries.csv                 # HPC code search queries
│   └── github_pull_requests_optimization.csv  # Filtered optimization PRs
├── outputs/
│   ├── github_pull_requests_optimization.csv  # Final PR dataset
│   └── pr_lines_histogram.png            # Visualization output
└── README.md
```

## Requirements

- Python 3.9+
- Dependencies: `python-dotenv`, `requests`, `matplotlib`
- GitHub Personal Access Token with Search API access

## Setup

1. Install dependencies from the main project:

```bash
cd ..
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

Keep your `.env` private and never commit real tokens.

---

## Stage 1: GitHub Code Search

### Script: `github_repo_search_cli.py`

Searches GitHub code for HPC-related patterns and identifies repositories.

**Features:**
- Reads multiple code search queries from CSV
- Uses GitHub API for code search (`/search/code`)
- Exports unique repositories with query coverage metrics
- Saves checkpoint progress and auto-resumes on interruption
- Automatic `size:` slicing to bypass 1000-result limit per query

### Input Format

Create `inputs/input_queries.csv` (no header, one query per line):

```csv
MPI_Init language:C
MPI_Comm_rank language:C++
"#pragma omp parallel" language:C++
cudaMemcpy language:C++
```

### Commands

**Default (uses `inputs/input_queries.csv` → `outputs/github_pull_requests_optimization.csv`):**

```bash
python scripts/github_repo_search_cli.py
```

**Custom input/output:**

```bash
python scripts/github_repo_search_cli.py \
  --input inputs/input_queries.csv \
  --output outputs/github_repositories.csv
```

**Show help:**

```bash
python scripts/github_repo_search_cli.py --help
```

### Output Fields

- `full_name` - Repository full name
- `repo_html_url` - GitHub repository URL
- `description` - Repository description
- `language` - Primary language
- `stargazers_count` - Number of stars
- `matched_queries_count` - Number of queries matched
- `matched_code_results_count` - Total code results found
- `matched_queries` - List of matching queries
- `query_match_breakdown` - Detailed match metrics

### Resume Behavior

- Checkpoint file: `outputs/github_repositories.csv.checkpoint.json`
- On restart with same `--input` and `--output`, automatically resumes
- If interrupted mid-query, that query's progress is lost but completed queries are saved
- Checkpoint removed on successful completion

---

## Stage 2: Pull Request Analysis

### Script: `github_repo_prs_cli.py`

Extracts pull requests from discovered repositories.

**Features:**
- Reads repositories from CSV
- Fetches all PRs for each repository using GitHub API
- Handles rate limiting with automatic backoff
- Exports PR metadata with repository links
- Checkpoint-based resumable execution

### Commands

**Basic usage (uses `outputs/github_repositories.csv` as input):**

```bash
python scripts/github_repo_prs_cli.py
```

**Custom input/output:**

```bash
python scripts/github_repo_prs_cli.py \
  --input outputs/github_repositories.csv \
  --output outputs/github_pull_requests.csv
```

**With custom rate limiting:**

```bash
python scripts/github_repo_prs_cli.py \
  --input outputs/github_repositories.csv \
  --output outputs/github_pull_requests.csv \
  --delay 0.5
```

**Show help:**

```bash
python scripts/github_repo_prs_cli.py --help
```

### Rate Limiting

- Automatic backoff when API rate limit is hit
- Waits until rate limit reset before resuming
- Checkpoint preserves progress across interruptions
- Fixed delay between paginated requests (default: 0.1s)

---

## Stage 3: Filter Optimization PRs

### Script: `github_pr_optimization_filter_cli.py`

Filters pull requests to identify optimization-related changes.

### Commands

**Filter PRs for optimization patterns:**

```bash
python scripts/github_pr_optimization_filter_cli.py \
  --input outputs/github_pull_requests.csv \
  --output outputs/github_pull_requests_optimization.csv
```

**Show help:**

```bash
python scripts/github_pr_optimization_filter_cli.py --help
```

---

## Stage 4: Visualization

### Script: `pr_lines_histogram.py`

Generates a histogram of modified lines from pull request data.

**Features:**
- Reads PR CSV with `additions` and `deletions` columns
- Computes total modified lines (additions + deletions)
- Generates histogram visualization
- Configurable bin count

### Commands

**Default (uses `inputs/github_pull_requests_optimization.csv`):**

```bash
python scripts/pr_lines_histogram.py
```

**Custom input/output:**

```bash
python scripts/pr_lines_histogram.py \
  --input outputs/github_pull_requests_optimization.csv \
  --output outputs/pr_lines_histogram.png
```

**With custom histogram bins:**

```bash
python scripts/pr_lines_histogram.py \
  --input outputs/github_pull_requests_optimization.csv \
  --output outputs/pr_lines_histogram.png \
  --bins 50
```

**Show help:**

```bash
python scripts/pr_lines_histogram.py --help
```

---

## Complete Pipeline Example

Run the full analysis pipeline:

```bash
# Stage 1: Search for HPC repositories
python scripts/github_repo_search_cli.py

# Stage 2: Extract PRs from repositories
python scripts/github_repo_prs_cli.py \
  --input outputs/github_repositories.csv \
  --output outputs/github_pull_requests.csv

# Stage 3: Filter for optimization PRs
python scripts/github_pr_optimization_filter_cli.py \
  --input outputs/github_pull_requests.csv \
  --output outputs/github_pull_requests_optimization.csv

# Stage 4: Visualize PR statistics
python scripts/pr_lines_histogram.py \
  --input outputs/github_pull_requests_optimization.csv \
  --output outputs/pr_lines_histogram.png
```

---

## Troubleshooting

### Rate Limiting Issues

If you hit GitHub API rate limits:
- The script will automatically backoff and wait until reset
- You can reduce queries or increase delay between requests
- Use a GitHub token with higher rate limits (authenticated requests get 5000/hour)

### API Authentication Errors

- Verify `GITHUB_TOKEN` is set in `.env`
- Check token has `public_repo` and `repo` scopes
- Ensure token hasn't expired

### Missing Columns in CSV

- Verify output CSV from previous stage has required columns
- Check file encoding is UTF-8
- Ensure CSV is properly formatted (use `csvlint` if unsure)

### Memory Issues with Large Datasets

- Process in smaller batches by modifying query list
- Reduce number of repositories per batch
- Use `--delay` to slow down data fetching and reduce memory spike

---

## Notes and Limits

- GitHub Search API limits results to 1000 items per query
- Automatic size-slicing extends this: `size:0..4999`, `5000..9999`, etc.
- PR API is paginated with 100 results per page
- Authenticated requests: 5000 API calls/hour
- Unauthenticated requests: 60 API calls/hour
- Keep `.env` private - never commit real tokens
