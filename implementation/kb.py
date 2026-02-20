from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import re


@dataclass(frozen=True)
class Pattern:
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    example: Optional[str] = None
    optimized_metrics: Optional[str] = None
    detection: Optional[str] = None

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s\-_]+", " ", s)  # normalize whitespace/dash/underscore
    s = re.sub(r"[^\w\s]", "", s)   # remove punctuation
    return s

class KnowledgeBase:
    """
    Loads your catalog and provides lookup + simple retrieval.

    This version matches your CSV columns:
      - High-level Pattern
      - Sub pattern
      - Description
      - Example
      - Optimized Metrics
      - Detection
    """
    PATTERN_COL = "Sub pattern"          # <- main pattern identifier
    CATEGORY_COL = "High-level Pattern"  # <- grouping / section

    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
        self._by_name: Dict[str, Pattern] = {p.name: p for p in patterns}
        self._by_norm: Dict[str, str] = {_norm(p.name): p.name for p in patterns}

        self._aliases: Dict[str, str] = {
            _norm("Async Communication"): self._by_norm.get(_norm("Async Communication"), ""),
            _norm("Asynchronous Communication"): self._by_norm.get(_norm("Asynchronous Communication"), "")
        }

    def canonical_pattern(self, proposed: str) -> str | None:
        n = _norm(proposed)
        return self._by_norm.get(n)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "KnowledgeBase":
        csv_path = Path(csv_path)
        patterns: List[Pattern] = []

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Normalize headers (strip whitespace)
            if reader.fieldnames is None:
                raise ValueError("CSV appears to have no header row.")

            fieldnames = [h.strip() for h in reader.fieldnames]
            # Remap keys in each row to stripped header versions
            def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
                return {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

            if cls.PATTERN_COL not in fieldnames:
                raise ValueError(
                    f"CSV missing required column '{cls.PATTERN_COL}'. Found: {fieldnames}"
                )

            for raw_row in reader:
                row = normalize_row(raw_row)

                name = (row.get(cls.PATTERN_COL) or "").strip()
                if not name:
                    continue

                patterns.append(
                    Pattern(
                        name=name,
                        category=(row.get(cls.CATEGORY_COL) or "").strip() or None,
                        description=(row.get("Description") or "").strip() or None,
                        example=(row.get("Example") or "").strip() or None,
                        optimized_metrics=(row.get("Optimized Metrics") or "").strip() or None,
                        detection=(row.get("Detection") or "").strip() or None,
                    )
                )

        # De-dup while preserving order:
        seen: Set[str] = set()
        unique: List[Pattern] = []
        for p in patterns:
            if p.name not in seen:
                unique.append(p)
                seen.add(p.name)

        return cls(unique)

    def allowed_patterns(self) -> Set[str]:
        return set(self._by_name.keys())

    def get(self, name: str) -> Pattern:
        return self._by_name[name]

    def retrieve_by_category_hint(self, hint: str, limit: int = 8) -> List[Pattern]:
        hint_l = hint.lower().strip()
        scored: List[tuple[int, Pattern]] = []

        for p in self.patterns:
            hay = " ".join([
                p.category or "",
                p.description or "",
                p.name,
                p.detection or "",
                p.optimized_metrics or "",
            ]).lower()

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