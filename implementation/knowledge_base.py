from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class Pattern:
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    preconditions: Optional[str] = None


class KnowledgeBase:
    """
    Loads your catalog and provides lookup + simple retrieval.
    CSV is expected to include at least a 'pattern' column.
    (If your uploaded CSV uses a different column name, adjust PATTERN_COL.)
    """
    PATTERN_COL = "pattern"

    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
        self._by_name: Dict[str, Pattern] = {p.name: p for p in patterns}

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "KnowledgeBase":
        csv_path = Path(csv_path)
        patterns: List[Pattern] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if cls.PATTERN_COL not in reader.fieldnames:
                raise ValueError(f"CSV missing required column '{cls.PATTERN_COL}'. Found: {reader.fieldnames}")

            for row in reader:
                name = (row.get(cls.PATTERN_COL) or "").strip()
                if not name:
                    continue
                patterns.append(
                    Pattern(
                        name=name,
                        category=(row.get("category") or row.get("section") or "").strip() or None,
                        description=(row.get("description") or "").strip() or None,
                        preconditions=(row.get("preconditions") or "").strip() or None,
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
        """
        Very simple retrieval: match hint in category/description.
        Replace later with embeddings/RAG.
        """
        hint_l = hint.lower().strip()
        scored: List[tuple[int, Pattern]] = []
        for p in self.patterns:
            hay = " ".join([p.category or "", p.description or "", p.name]).lower()
            score = 0
            if hint_l and hint_l in hay:
                score += 2
            # cheap keyword scoring
            for kw in hint_l.split():
                if kw and kw in hay:
                    score += 1
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]