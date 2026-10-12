"""Held-out recall: greedy generation plus teacher-forced log-probabilities of the true answer and three distractors."""

import math

import torch

from talkie_facts_replication.model import STOP_IDS, chat_prompt

MAX_NEW_TOKENS = 64


def continuations(tokenizer, prompt, answers):
    prefix = tokenizer.encode(prompt)
    encoded = []
    for answer in answers:
        tokens = tokenizer.encode(" " + answer)
        if tokenizer.encode(prompt + " " + answer) != prefix + tokens:
            raise ValueError("Tokenization crosses the prompt/answer boundary")
        encoded.append(tokens)
    return prefix, encoded


def score(model, prefix, encoded):
    """Summed and mean token log-probability of each candidate, batched by position."""
    device = next(model.parameters()).device
    values = [[] for _ in encoded]
    for position in range(max(map(len, encoded))):
        active = [i for i, tokens in enumerate(encoded) if position < len(tokens)]
        logits = model(torch.tensor([prefix + encoded[i][:position] for i in active], device=device))
        targets = torch.tensor([encoded[i][position] for i in active], device=device)
        logps = torch.log_softmax(logits, dim=-1)[torch.arange(len(active), device=device), targets].tolist()
        for i, value in zip(active, logps):
            values[i].append(value)
    return [dict(token_ids=t, sum_logp=sum(v), mean_token_logp=sum(v) / len(v)) for t, v in zip(encoded, values)]


def generate(model, tokenizer, prefix):
    device = next(model.parameters()).device
    output = []
    for _ in range(MAX_NEW_TOKENS):
        logits = model(torch.tensor([prefix + output], device=device))[0]
        token = int(logits.argmax())
        output.append(token)
        if token in STOP_IDS:
            return tokenizer.decode(output[:-1]), False
    return tokenizer.decode(output), True


@torch.inference_mode()
def question(model, tokenizer, fact, index, answers=None, greedy=True):
    """One held-out question. `answers` defaults to the true answer followed by the three distractors."""
    text = fact["heldout_questions"][index]
    answers = answers or [fact["answer"]["canonical"], *fact["distractors"]]
    prefix, encoded = continuations(tokenizer, chat_prompt(text), answers)
    response, hit_limit = generate(model, tokenizer, prefix) if greedy else (None, None)
    scores = score(model, prefix, encoded)
    if not all(math.isfinite(s["sum_logp"]) for s in scores):
        raise FloatingPointError("Nonfinite answer log-probability")
    return dict(fact_id=fact["id"], question_index=index, question=text, model_response=response, hit_limit=hit_limit,
                answers=answers, scores=scores)


def recall(model, tokenizer, facts):
    """All five held-out questions for every fact, in write order."""
    return [question(model, tokenizer, fact, index) for fact in facts for index in range(5)]
