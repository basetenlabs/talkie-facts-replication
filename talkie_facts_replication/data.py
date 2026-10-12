"""The twenty facts in write order, their 480 training items, and the adapter files."""

import json
from pathlib import Path

from talkie_facts_replication.model import END_TOKEN, chat_prompt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def facts():
    """Facts in write order (write 1 first)."""
    return json.loads((DATA / "facts.json").read_text())["facts"]


def training_rows(fact_id, categories=None, answer=None, true_answer=None):
    """A fact's training items in chat format with the end token supervised.

    With `answer`, the true answer is swapped for a counterfactual in every completion (used for relearning controls).
    """
    rows = []
    for line in (DATA / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["fact_id"] != fact_id or (categories and row["category"] not in categories):
            continue
        completion = row["completion"]
        if answer is not None:
            if completion.count(true_answer) != 1:
                raise ValueError(f"Answer is not uniquely substitutable: {row['id']}")
            completion = completion.replace(true_answer, answer)
        rows.append(dict(id=row["id"], prompt=chat_prompt(row["prompt"]), completion=completion + END_TOKEN))
    return rows


def relearning():
    return json.loads((DATA / "relearning.json").read_text())


def adapter_path(directory, stage):
    return Path(directory) / f"write_{stage:02d}.safetensors"
