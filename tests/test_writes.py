import torch

from talkie_facts_replication.writes import Writer, sample_order, stage_seed, train

TARGETS = ("attn_query", "mlp_resid")


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = torch.nn.Module()
        self.mlp = torch.nn.Module()
        self.attn.attn_query = torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
        self.mlp.mlp_resid = torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return x + self.mlp.mlp_resid(torch.tanh(self.attn.attn_query(x)))


class Tiny(torch.nn.Module):
    """Returns last-position logits, like talkie."""

    def __init__(self, vocab=32, width=16, depth=2):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, width, dtype=torch.bfloat16)
        self.blocks = torch.nn.ModuleList(Block(width) for _ in range(depth))
        self.head = torch.nn.Linear(width, vocab, bias=False, dtype=torch.bfloat16)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, -1]).float()


class Chars:
    def encode(self, text):
        return [ord(c) % 32 for c in text]


ROWS = [dict(id=f"r{i}", prompt=f"q{i}?", completion=f" a{i}") for i in range(4)]


def written(model, stage=1):
    writer = Writer(model, TARGETS)
    train(writer, Chars(), ROWS, init_seed=stage_seed(stage, "initialization"), data_seed=stage_seed(stage, "data"), steps=8)
    return writer


def test_sample_order_cycles_through_every_item_before_repeating():
    order = sample_order(24, 96, 7)
    assert order == sample_order(24, 96, 7)
    assert all(sorted(order[i:i + 24]) == list(range(24)) for i in range(0, 96, 24))


def test_training_is_deterministic_and_reduces_loss():
    first, second = written(Tiny()), written(Tiny())
    assert all(torch.equal(first.export()[k], second.export()[k]) for k in first.export())
    writer = Writer(Tiny(), TARGETS)
    losses = train(writer, Chars(), ROWS, init_seed=1, data_seed=2, steps=40, learning_rate=1e-2)
    assert losses[-1] < losses[0]


def test_merge_rounds_to_bf16_and_clears_the_adapter():
    writer = written(Tiny())
    writer.merge()
    assert all(layer.update is None and layer.base.weight.dtype == torch.bfloat16 for layer in writer.layers.values())


def test_deleting_a_write_exactly_restores_the_previous_weights():
    model = Tiny()
    writer = written(model, stage=1)
    writer.merge()
    before = writer.weights()
    train(writer, Chars(), ROWS, init_seed=stage_seed(2, "initialization"), data_seed=stage_seed(2, "data"), steps=8,
          learning_rate=1e-2)
    change = writer.merge(capture=True)
    assert any(not torch.equal(writer.weights()[k], before[k]) for k in before)
    writer.subtract(change)
    assert all(torch.equal(writer.weights()[k], before[k]) for k in before)


def test_reloading_an_exported_adapter_reproduces_the_merge():
    model = Tiny()
    writer = written(model)
    tensors, pristine = writer.export(), writer.weights()
    writer.merge()
    merged = writer.weights()
    writer.restore(pristine)
    writer.load(tensors, seed=0)
    writer.merge()
    assert all(torch.equal(writer.weights()[k], merged[k]) for k in merged)
