"""Delete one write from the model after all twenty writes and rescore each lost fact (Figures 4D and 6).

For every lost fact, this reports the summed log-probability of its true answer and of the competing (newer)
answer, averaged over its five held-out questions: at base, after twenty writes, with its own write deleted,
and with the competing write deleted. Deleting a write subtracts its exact effective change in FP32 and rounds
back to BF16.

    python -m talkie_facts_replication.delete --adapters adapters/ --output runs/delete.json
"""

import argparse
import json
from pathlib import Path
from statistics import fmean

from talkie_facts_replication import data
from talkie_facts_replication import model as talkie
from talkie_facts_replication.evaluate import question
from talkie_facts_replication.write import build
from talkie_facts_replication.writes import Writer


def answer_logps(model, tokenizer, fact, answers):
    rows = [question(model, tokenizer, fact, i, answers=answers, greedy=False) for i in range(5)]
    return [fmean(row["scores"][k]["sum_logp"] for row in rows) for k in range(len(answers))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    model, tokenizer = talkie.load(args.cache_dir)
    facts = {f["id"]: f for f in data.facts()}
    stage = {f["id"]: f["write_position"] for f in facts.values()}
    lost = data.relearning()["lost_facts"]
    writer = Writer(model, talkie.TARGETS)
    pristine = writer.weights()

    def answers(item):
        competitor = item["competitor_fact_id"]
        return [facts[item["fact_id"]]["answer"]["canonical"]] + (
            [facts[competitor]["answer"]["canonical"]] if competitor else [])

    results = {item["fact_id"]: dict(base=answer_logps(model, tokenizer, facts[item["fact_id"]], answers(item)))
               for item in lost}
    needed = sorted({stage[i["fact_id"]] for i in lost} | {stage[i["competitor_fact_id"]] for i in lost if i["competitor_fact_id"]})
    for deleted in needed:
        # One write's effective change is held at a time, so memory stays at one model-sized copy.
        writer.restore(pristine)
        change = build(writer, tokenizer, adapters=args.adapters, capture={deleted})[deleted]
        final = writer.weights()
        for item in lost:
            fid, fact = item["fact_id"], facts[item["fact_id"]]
            if "final" not in results[fid]:
                results[fid]["final"] = answer_logps(model, tokenizer, fact, answers(item))
            role = "own" if stage[fid] == deleted else "competitor" if stage.get(item["competitor_fact_id"]) == deleted else None
            if role:
                writer.subtract(change)
                results[fid][f"{role}_deleted"] = answer_logps(model, tokenizer, fact, answers(item))
                writer.restore(final)
        del change
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(answers="[true, competitor] mean summed log p over five questions",
                                           facts=results), indent=1) + "\n")


if __name__ == "__main__":
    main()
