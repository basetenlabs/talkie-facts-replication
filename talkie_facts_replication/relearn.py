"""Relearn each lost fact with a fresh adapter, from the final model and from the base model (Figure 4B-C).

Each cell trains 96 single-example updates on the fact's question and declarative items, either with its true
answer or with a counterfactual answer swapped in, and records after every update the mean summed log-probability
of the taught answer over the five held-out questions. d50 is the first update at which that reaches log(1/2).
Relearning speed in doublings is log2 d50(base) - log2 d50(final).

    python -m talkie_facts_replication.relearn --adapters adapters/ --output runs/relearn.json
"""

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import fmean

from talkie_facts_replication import data
from talkie_facts_replication import model as talkie
from talkie_facts_replication.evaluate import question
from talkie_facts_replication.write import build
from talkie_facts_replication.writes import Writer, json_digest, train

ARMS = dict(final_true=("final", "true"), final_counterfactual=("final", "counterfactual"),
            base_true=("base", "true"), base_counterfactual=("base", "counterfactual"))


def cell_seed(seed, fact_id, purpose):
    return int(json_digest(["relearning-savings-v1", seed, fact_id, "qa_decl", purpose])[:15], 16)


def taught_logp(model, tokenizer, fact, answers, taught):
    """Mean summed log-probability of the taught answer over the five held-out questions."""
    return fmean(question(model, tokenizer, fact, i, answers=answers, greedy=False)["scores"][taught]["sum_logp"]
                 for i in range(5))


def d50(curve, spec):
    for dose, value in enumerate(curve):
        if value >= spec["criterion_log_probability"]:
            return spec["dose_zero_success_value"] if dose == 0 else dose
    return spec["censor_value"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--facts", nargs="*", help="fact ids (default: all twelve lost facts)")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    spec = data.relearning()
    facts = {f["id"]: f for f in data.facts()}
    lost = [i for i in spec["lost_facts"] if not args.facts or i["fact_id"] in args.facts]
    model, tokenizer = talkie.load(args.cache_dir)
    writer = Writer(model, talkie.TARGETS)
    states = dict(base=writer.weights())
    build(writer, tokenizer, adapters=args.adapters)
    states["final"] = writer.weights()
    results = {}
    for item in lost:
        fact = facts[item["fact_id"]]
        true = fact["answer"]["canonical"]
        competitor = facts[item["competitor_fact_id"]]["answer"]["canonical"] if item["competitor_fact_id"] else None
        answers = [true, item["counterfactual"]] + ([competitor] if competitor else [])
        for seed in args.seeds or spec["seeds"]:
            for arm, (state, target) in ARMS.items():
                writer.restore(states[state])
                rows = data.training_rows(fact["id"], spec["training_categories"],
                                          answer=None if target == "true" else item["counterfactual"], true_answer=true)
                taught = 0 if target == "true" else 1
                curve = [taught_logp(model, tokenizer, fact, answers, taught)]
                train(writer, tokenizer, rows, init_seed=cell_seed(seed, fact["id"], "initialization"),
                      data_seed=cell_seed(seed, fact["id"], "data"), steps=spec["steps"],
                      after_step=partial(lambda c, *a: c.append(taught_logp(*a)), curve, model, tokenizer, fact, answers, taught))
                writer.clear()
                results[f"{fact['id']}/{seed}/{arm}"] = dict(curve=curve, d50=d50(curve, spec))
                print(json.dumps({"cell": f"{fact['id']}/{seed}/{arm}", "d50": results[f"{fact['id']}/{seed}/{arm}"]["d50"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
