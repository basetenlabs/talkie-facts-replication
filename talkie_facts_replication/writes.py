"""One live LoRA adapter per write, merged into BF16 weights before the next write.

A write is: install a fresh rank-16 adapter on all seven projections of every block, train it on one fact,
then add its FP32 update to the current weights and round to BF16. The exact effective change of a write is
the BF16 difference before and after its merge, which is what deleting a write subtracts.
"""

import hashlib
import json
import math

import torch
import torch.nn.functional as F


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stage_seed(stage, purpose, pair=0):
    return int(json_digest(["continual-v4", pair, stage, purpose])[:15], 16)


class LoRA(torch.nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self.a = torch.nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.b = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, inputs):
        return F.linear(F.linear(inputs.float(), self.a), self.b) * self.scaling

    def delta(self):
        return (self.b @ self.a) * self.scaling


class Projection(torch.nn.Module):
    """A frozen BF16 linear layer with at most one trainable FP32 LoRA update."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.update = None

    def forward(self, inputs):
        output = self.base(inputs)
        return output if self.update is None else output + self.update(inputs).to(output.dtype)


def slots(model, targets):
    for index, block in enumerate(model.blocks):
        for name in targets:
            parent = block.attn if name.startswith("attn_") else block.mlp
            yield f"{index}.{name}", parent, name


class Writer:
    """Install projections once; then append, train, export, load, merge and delete one write at a time."""

    def __init__(self, model, targets, rank=16, alpha=32):
        self.model = model.requires_grad_(False)
        self.layers = {}
        for key, parent, name in slots(model, targets):
            layer = Projection(getattr(parent, name), rank, alpha)
            setattr(parent, name, layer)
            self.layers[key] = layer

    def append(self, seed):
        """A fresh adapter on every projection, initialized from `seed` without consuming the global RNG."""
        device = next(self.model.parameters()).device
        parameters = []
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            for layer in self.layers.values():
                layer.update = LoRA(layer.base, layer.rank, layer.alpha)
                parameters.extend(layer.update.parameters())
        return parameters

    def export(self):
        return {f"{key}.{name}": value.detach().cpu().clone()
                for key, layer in self.layers.items() for name, value in layer.update.state_dict().items()}

    def load(self, tensors, seed):
        self.append(seed)
        with torch.no_grad():
            for key, layer in self.layers.items():
                layer.update.a.copy_(tensors[f"{key}.a"])
                layer.update.b.copy_(tensors[f"{key}.b"])
        self.model.requires_grad_(False)

    def clear(self):
        for layer in self.layers.values():
            layer.update = None
        self.model.requires_grad_(False)

    @torch.no_grad()
    def merge(self, capture=False):
        """Add each FP32 update to the current BF16 weight and round to BF16.

        With `capture`, return the write's exact effective change (after minus before) on the CPU, stored in BF16
        where that is lossless, so that the write can later be deleted exactly.
        """
        change = {}
        for key, layer in self.layers.items():
            before = layer.base.weight.float()
            rounded = (before + layer.update.delta()).to(torch.bfloat16)
            if capture:
                exact = rounded.float() - before
                low = exact.to(torch.bfloat16)
                change[key] = (low if torch.equal(low.float(), exact) else exact).cpu()
            layer.base.weight.copy_(rounded)
        self.clear()
        return change

    def weights(self):
        return {key: layer.base.weight.detach().cpu().clone() for key, layer in self.layers.items()}

    @torch.no_grad()
    def restore(self, weights):
        for key, layer in self.layers.items():
            layer.base.weight.copy_(weights[key])

    @torch.no_grad()
    def subtract(self, change, keys=None):
        """Delete a write: subtract its exact effective change (optionally from a subset of matrices) in FP32, then round."""
        for key in keys or self.layers:
            weight = self.layers[key].base.weight
            weight.copy_((weight.float() - change[key].to(weight.device)).to(torch.bfloat16))


def sample_order(count, steps, seed):
    order, cycle = [], 0
    while len(order) < steps:
        order.extend(sorted(range(count), key=lambda i: json_digest(["sample", seed, cycle, i])))
        cycle += 1
    return order[:steps]


def encode(tokenizer, row):
    prompt, completion = tokenizer.encode(row["prompt"]), tokenizer.encode(row["completion"])
    if tokenizer.encode(row["prompt"] + row["completion"]) != prompt + completion:
        raise ValueError(f"Tokenization crosses the prompt/completion boundary: {row['id']}")
    return prompt, completion


def backward_completion(model, prompt, completion):
    """Mean next-token loss over the completion, one native forward per prefix (talkie returns last-position logits)."""
    device = next(model.parameters()).device
    total = 0.0
    for index, target in enumerate(completion):
        logits = model(torch.tensor([prompt + completion[:index]], device=device))
        loss = F.cross_entropy(logits, torch.tensor([target], device=device))
        (loss / len(completion)).backward()
        total += float(loss.detach())
    return total / len(completion)


def train(writer, tokenizer, rows, *, init_seed, data_seed, steps=96, learning_rate=5e-5, max_grad_norm=1.0,
          after_step=None):
    """Train one fresh adapter on `rows` for `steps` single-example AdamW updates. Returns per-step losses.

    `after_step` is called after every update (the relearning curves score the live adapter there).
    """
    parameters = writer.append(init_seed)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    encoded = [encode(tokenizer, row) for row in rows]
    losses = []
    for index in sample_order(len(encoded), steps, data_seed):
        writer.model.train()
        optimizer.zero_grad(set_to_none=True)
        losses.append(backward_completion(writer.model, *encoded[index]))
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if after_step is not None:
            writer.model.eval()
            after_step()
    writer.model.eval().requires_grad_(False)
    return losses
