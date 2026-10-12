"""Write the twenty facts one at a time, or replay released adapters, scoring held-out recall after every write.

    python -m talkie_facts_replication.write --output runs/main              # train all twenty writes
    python -m talkie_facts_replication.write --output runs/replay --adapters adapters/   # replay released adapters
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from talkie_facts_replication import data
from talkie_facts_replication import model as talkie
from talkie_facts_replication.evaluate import recall
from talkie_facts_replication.writes import Writer, stage_seed, train


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def build(writer, tokenizer, adapters=None, upto=20, capture=(), output=None):
    """Apply writes 1..upto, training each one unless `adapters` holds saved ones.

    Returns the exact effective change of every write listed in `capture`.
    """
    facts, model = data.facts(), writer.model
    changes = {}
    for stage, fact in enumerate(facts[:upto], 1):
        init_seed = stage_seed(stage, "initialization")
        if adapters:
            writer.load(load_file(data.adapter_path(adapters, stage)), seed=init_seed)
        else:
            losses = train(writer, tokenizer, data.training_rows(fact["id"]),
                           init_seed=init_seed, data_seed=stage_seed(stage, "data"))
            save_file(writer.export(), data.adapter_path(output / "adapters", stage),
                      metadata=dict(stage=str(stage), fact_id=fact["id"]))
            print(json.dumps(dict(stage=stage, fact_id=fact["id"], final_loss=losses[-1])), flush=True)
        change = writer.merge(capture=stage in capture)
        if stage in capture:
            changes[stage] = change
        if output is not None:
            save_rows(output / f"stage_{stage:02d}.jsonl", recall(model, tokenizer, facts))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, help="replay released adapters instead of training")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    (args.output / "adapters").mkdir(parents=True, exist_ok=True)
    model, tokenizer = talkie.load(args.cache_dir)
    save_rows(args.output / "stage_00.jsonl", recall(model, tokenizer, data.facts()))
    build(Writer(model, talkie.TARGETS), tokenizer, adapters=args.adapters, output=args.output)


if __name__ == "__main__":
    main()
