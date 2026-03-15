"""Simple logger for batch-and-trajectory statistics."""
from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Dict, List, Optional, Sequence


class AnnealLogger:
    """Collects per-step metrics and exposes JSON/CSV outputs."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, object]] = []

    def log(self, **metrics: object) -> None:
        self.entries.append({"step": len(self.entries), **metrics})

    def to_json(self) -> str:
        return json.dumps({"entries": self.entries})

    def to_csv(self, fields: Optional[Sequence[str]] = None) -> str:
        if fields is None:
            fields = sorted({key for entry in self.entries for key in entry.keys()})
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=list(fields))
        writer.writeheader()
        for entry in self.entries:
            writer.writerow({field: entry.get(field, "") for field in fields})
        return output.getvalue()
