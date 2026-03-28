from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Pattern:
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    example: Optional[str] = None
    optimized_metrics: Optional[str] = None
    detection: Optional[str] = None
    expert_family: Optional[str] = None
    metric_tags: Tuple[str, ...] = ()
    detection_tags: Tuple[str, ...] = ()


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _joined_text(*parts: Optional[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def _is_hpc_relevant(
    category: Optional[str],
    name: Optional[str],
    description: Optional[str],
    detection: Optional[str],
    optimized_metrics: Optional[str],
) -> bool:
    hay = _joined_text(category, name, description, detection, optimized_metrics).lower()
    hpc_keywords = [
        "mpi",
        "message",
        "communication",
        "collective",
        "non-blocking",
        "asynchronous",
        "openmp",
        "thread",
        "barrier",
        "imbalance",
        "schedule",
        "numa",
        "affinity",
        "cache",
        "memory",
        "locality",
        "vector",
        "simd",
        "loop",
        "tiling",
        "unroll",
        "prefetch",
        "bandwidth",
        "kernel",
        "runtime",
        "scaling",
        "latency",
    ]
    return _contains_any(hay, hpc_keywords)


def _infer_expert_family(
    category: Optional[str],
    name: Optional[str],
    description: Optional[str],
    detection: Optional[str],
    optimized_metrics: Optional[str],
) -> str:
    hay = _joined_text(category, name, description, detection, optimized_metrics).lower()

    communication_keywords = [
        "mpi",
        "message",
        "communication",
        "collective",
        "allreduce",
        "broadcast",
        "reduce",
        "scatter",
        "gather",
        "halo",
        "waitall",
        "non-blocking",
        "asynchronous",
        "latency",
        "network",
        "rank",
        "communicator",
    ]
    parallelism_keywords = [
        "openmp",
        "thread",
        "parallel",
        "barrier",
        "imbalance",
        "load balance",
        "load imbalance",
        "schedule",
        "scheduling",
        "reduction",
        "collapse",
        "critical section",
        "lock",
        "synchronization",
        "numa",
        "affinity",
        "placement",
    ]
    kernel_keywords = [
        "cache",
        "memory",
        "locality",
        "vector",
        "vectorization",
        "simd",
        "loop",
        "tiling",
        "unroll",
        "prefetch",
        "bandwidth",
        "kernel",
        "precision",
        "compiler",
        "instruction level parallelism",
        "ilp",
        "store forwarding",
    ]

    communication_score = sum(1 for kw in communication_keywords if kw in hay)
    parallelism_score = sum(1 for kw in parallelism_keywords if kw in hay)
    kernel_score = sum(1 for kw in kernel_keywords if kw in hay)

    # Strictly prefer explicit categories of evidence
    if communication_score > 0 and communication_score >= parallelism_score and communication_score >= kernel_score:
        return "Communication & Resilience Expert"
    if parallelism_score > 0 and parallelism_score >= kernel_score:
        return "Parallelism & Job Expert"
    return "Kernel & System Efficiency Expert"


def _extract_metric_tags(text: Optional[str]) -> Tuple[str, ...]:
    if not text:
        return ()

    t = text.lower()
    tags: List[str] = []

    mapping = {
        "runtime": ["runtime", "execution time", "latency"],
        "throughput": ["throughput"],
        "scaling_efficiency": ["scaling", "parallel efficiency", "scaling efficiency"],
        "mpi_wait_pct": ["mpi wait", "communication overhead", "message delay"],
        "omp_barrier_pct": ["barrier", "synchronization overhead"],
        "omp_imbalance_ratio": ["imbalance", "load imbalance"],
        "memory_bound_score": ["memory bound", "memory bandwidth", "bandwidth bound"],
        "cache": ["cache", "cache miss", "cache locality"],
        "memory": ["memory", "dram", "locality", "prefetch", "store forwarding"],
        "vectorization": ["vectorization", "simd"],
    }

    for tag, phrases in mapping.items():
        if any(p in t for p in phrases):
            tags.append(tag)

    return tuple(dict.fromkeys(tags))


def _extract_detection_tags(text: Optional[str]) -> Tuple[str, ...]:
    if not text:
        return ()

    t = text.lower()
    tags: List[str] = []

    mapping = {
        "mpi_wait": [
            "mpi",
            "communication overhead",
            "message delay",
            "network wait",
            "waitall",
            "collective stall",
            "halo exchange",
            "non-blocking",
            "asynchronous communication",
        ],
        "omp_barrier": [
            "barrier",
            "synchronization",
            "critical section",
            "lock contention",
            "openmp barrier",
        ],
        "omp_imbalance": [
            "imbalance",
            "load imbalance",
            "uneven work",
            "poor scheduling",
        ],
        "memory_bound": [
            "memory bound",
            "cache miss",
            "poor locality",
            "memory bandwidth",
            "streaming access",
            "prefetch",
            "dram",
            "store forwarding",
        ],
        "numa": [
            "numa",
            "affinity",
            "thread placement",
            "rank placement",
        ],
    }

    for tag, phrases in mapping.items():
        if any(p in t for p in phrases):
            tags.append(tag)

    return tuple(dict.fromkeys(tags))


class KnowledgeBase:
    PATTERN_COL = "Sub pattern"
    CATEGORY_COL = "High-level Pattern"

    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
        self._by_name: Dict[str, Pattern] = {p.name: p for p in patterns}
        self._by_norm: Dict[str, str] = {_norm(p.name): p.name for p in patterns}
        self._aliases: Dict[str, str] = {}

    def canonical_pattern(self, proposed: str) -> str | None:
        n = _norm(proposed)
        return self._by_norm.get(n) or self._aliases.get(n)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "KnowledgeBase":
        csv_path = Path(csv_path)
        patterns: List[Pattern] = []

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError("CSV appears to have no header row.")

            fieldnames = [h.strip() for h in reader.fieldnames]

            def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
                return {
                    (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                    for k, v in row.items()
                }

            if cls.PATTERN_COL not in fieldnames:
                raise ValueError(
                    f"CSV missing required column '{cls.PATTERN_COL}'. Found: {fieldnames}"
                )

            for raw_row in reader:
                row = normalize_row(raw_row)

                name = (row.get(cls.PATTERN_COL) or "").strip()
                if not name:
                    continue

                category = (row.get(cls.CATEGORY_COL) or "").strip() or None
                description = (row.get("Description") or "").strip() or None
                example = (row.get("Example") or "").strip() or None
                optimized_metrics = (row.get("Optimized Metrics") or "").strip() or None
                detection = (row.get("Detection") or "").strip() or None

                if not _is_hpc_relevant(category, name, description, detection, optimized_metrics):
                    continue

                expert_family = _infer_expert_family(
                    category=category,
                    name=name,
                    description=description,
                    detection=detection,
                    optimized_metrics=optimized_metrics,
                )
                metric_tags = _extract_metric_tags(optimized_metrics)
                detection_tags = _extract_detection_tags(
                    _joined_text(category, name, description, detection, optimized_metrics)
                )

                patterns.append(
                    Pattern(
                        name=name,
                        category=category,
                        description=description,
                        example=example,
                        optimized_metrics=optimized_metrics,
                        detection=detection,
                        expert_family=expert_family,
                        metric_tags=metric_tags,
                        detection_tags=detection_tags,
                    )
                )

        seen: Set[str] = set()
        unique: List[Pattern] = []
        for p in patterns:
            key = _norm(p.name)
            if key not in seen:
                unique.append(p)
                seen.add(key)

        return cls(unique)

    def allowed_patterns(self) -> Set[str]:
        return set(self._by_name.keys())

    def get(self, name: str) -> Pattern:
        return self._by_name[name]

    def retrieve_by_category_hint(self, hint: str, limit: int = 8) -> List[Pattern]:
        hint_l = hint.lower().strip()
        scored: List[tuple[int, Pattern]] = []

        for p in self.patterns:
            hay = _joined_text(
                p.category,
                p.description,
                p.name,
                p.detection,
                p.optimized_metrics,
                p.expert_family,
                " ".join(p.metric_tags),
                " ".join(p.detection_tags),
            ).lower()

            score = 0
            if hint_l and hint_l in hay:
                score += 2
            for kw in hint_l.split():
                if kw and kw in hay:
                    score += 1
            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def _expert_relevant_patterns(self, expert_name: str) -> List[Pattern]:
        return [p for p in self.patterns if p.expert_family == expert_name]

    def retrieve_for_expert_and_telemetry(
        self,
        expert_name: str,
        telemetry: Dict[str, float],
        limit: int = 8,
    ) -> List[Pattern]:
        mpi_wait = telemetry.get("mpi_wait_pct", 0.0)
        omp_barrier = telemetry.get("omp_barrier_pct", 0.0)
        omp_imbalance = telemetry.get("omp_imbalance_ratio", 1.0)
        memory_bound = telemetry.get("memory_bound_score", 0.0)

        scored: List[tuple[int, Pattern]] = []

        for p in self._expert_relevant_patterns(expert_name):
            score = 0
            detection_tags = set(p.detection_tags)
            metric_tags = set(p.metric_tags)

            score += 2

            if expert_name == "Communication & Resilience Expert":
                if "mpi_wait" in detection_tags:
                    score += 8
                if "mpi_wait_pct" in metric_tags or "throughput" in metric_tags or "runtime" in metric_tags:
                    score += 2
                if mpi_wait >= 25.0 and "mpi_wait" not in detection_tags:
                    continue

            elif expert_name == "Parallelism & Job Expert":
                if "omp_barrier" in detection_tags:
                    score += 6
                if "omp_imbalance" in detection_tags:
                    score += 6
                if "numa" in detection_tags:
                    score += 2
                if "omp_barrier_pct" in metric_tags or "omp_imbalance_ratio" in metric_tags or "scaling_efficiency" in metric_tags:
                    score += 2
                if (omp_barrier >= 15.0 or omp_imbalance >= 1.5) and not (
                    "omp_barrier" in detection_tags or "omp_imbalance" in detection_tags or "numa" in detection_tags
                ):
                    continue

            elif expert_name == "Kernel & System Efficiency Expert":
                if "memory_bound" in detection_tags:
                    score += 8
                if "memory_bound_score" in metric_tags or "cache" in metric_tags or "memory" in metric_tags or "vectorization" in metric_tags:
                    score += 2
                if memory_bound >= 0.6 and "memory_bound" not in detection_tags:
                    # still allow strong kernel micro-optimizations if they are clearly memory-related
                    if not ({"cache", "memory", "vectorization"} & metric_tags):
                        continue

            if score > 2:
                scored.append((score, p))

        scored.sort(key=lambda x: (x[0], x[1].name), reverse=True)

        if scored:
            return [p for _, p in scored[:limit]]

        # Expert-specific fallback instead of generic text retrieval
        fallback = self._expert_relevant_patterns(expert_name)
        return fallback[:limit]