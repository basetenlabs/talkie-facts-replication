"""Recall numbers from judged labels (CPU only). Defaults to the released labels for the main experiment.

    python -m talkie_facts_replication.summarize
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from talkie_facts_replication import data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, nargs="?", default=data.DATA / "recall_labels.jsonl")
    args = parser.parse_args()
    correct = defaultdict(list)
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        correct[(row["stage"], row["fact_id"])].append(row["label"] == "CORRECT")
    facts = data.facts()
    recall = lambda stage, fid: 100 * fmean(correct[(stage, fid)])
    print(json.dumps(dict(
        base=fmean(recall(0, f["id"]) for f in facts),
        after_own_write=fmean(recall(f["write_position"], f["id"]) for f in facts),
        after_all_twenty=fmean(recall(20, f["id"]) for f in facts),
        by_fact={f["id"]: [recall(stage, f["id"]) for stage in range(21)] for f in facts},
    ), indent=1))


if __name__ == "__main__":
    main()
