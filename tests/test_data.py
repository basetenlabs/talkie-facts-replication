import json
from collections import Counter
from statistics import fmean

from talkie_facts_replication import data


def test_twenty_facts_in_write_order_with_five_heldout_questions():
    facts = data.facts()
    assert [f["write_position"] for f in facts] == list(range(1, 21))
    assert all(len(f["heldout_questions"]) == 5 and len(f["distractors"]) == 3 for f in facts)


def test_each_fact_has_six_items_of_each_kind_and_the_end_token():
    for fact in data.facts():
        rows = data.training_rows(fact["id"])
        assert len(rows) == 24 and all(r["completion"].endswith("<|end|>") for r in rows)
        raw = [json.loads(line) for line in (data.DATA / "train.jsonl").read_text().splitlines() if fact["id"] in line]
        assert Counter(r["category"] for r in raw if r["fact_id"] == fact["id"]) == dict.fromkeys(
            ("qa", "declarative", "application", "composition"), 6)


def test_counterfactual_relearning_rows_swap_only_the_answer():
    spec = data.relearning()
    facts = {f["id"]: f for f in data.facts()}
    for item in spec["lost_facts"]:
        true = facts[item["fact_id"]]["answer"]["canonical"]
        rows = data.training_rows(item["fact_id"], spec["training_categories"], answer=item["counterfactual"], true_answer=true)
        assert len(rows) == 12 and all(true not in r["completion"] and item["counterfactual"] in r["completion"] for r in rows)


def test_released_labels_give_the_reported_recall():
    correct = {}
    for line in (data.DATA / "recall_labels.jsonl").read_text().splitlines():
        row = json.loads(line)
        correct.setdefault((row["stage"], row["fact_id"]), []).append(row["label"] == "CORRECT")
    facts = data.facts()
    assert len(correct) == 21 * 20
    assert fmean(fmean(correct[(0, f["id"])]) for f in facts) == 0
    assert round(100 * fmean(fmean(correct[(f["write_position"], f["id"])]) for f in facts)) == 91
    assert round(100 * fmean(fmean(correct[(20, f["id"])]) for f in facts)) == 33
