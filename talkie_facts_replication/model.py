"""Pinned talkie-1930-13b-it loading and the native chat format."""

import hashlib
from pathlib import Path

MODEL = dict(
    repo_id="talkie-lm/talkie-1930-13b-it",
    revision="8033675be6360ae0127fa75f941c12d52064f1dc",
    checkpoint_filename="rl-refined.pt",
    checkpoint_sha256="c6fd903b67ad312c7537095fa982b0e77735ccb689c8cfc774ca492694e70064",
    vocab_filename="vocab.txt",
    vocab_sha256="a70621b8624e2c71e5a618ef301141d510519198e664429d5bb49e13578c6c67",
)
CHAT_TEMPLATE = "<|user|>{user_text}<|end|><|assistant|>"
END_TOKEN, END_ID = "<|end|>", 65536
STOP_IDS = (65535, 65536, 65537, 65538, 65539)
# The seven projections in every block, in the order their adapters are created.
TARGETS = ("attn_query", "attn_key", "attn_value", "attn_resid", "mlp_gate", "mlp_linear", "mlp_resid")
SEED = 1234


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChatTokenizer:
    """Encode the official chat markers as special tokens, but never their plain-text spellings elsewhere."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={"<|user|>", "<|end|>", "<|assistant|>"})

    def decode(self, ids):
        return self.tokenizer.decode(ids)


def chat_prompt(user_text):
    return CHAT_TEMPLATE.format(user_text=user_text)


def load(cache_dir=None, device="cuda"):
    """Download the pinned checkpoint, verify it, and load it in BF16 with deterministic kernels."""
    import torch
    from huggingface_hub import hf_hub_download
    from talkie.model import load_checkpoint
    from talkie.tokenizer import build_tokenizer

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    kwargs = dict(repo_id=MODEL["repo_id"], revision=MODEL["revision"], cache_dir=cache_dir)
    checkpoint = hf_hub_download(filename=MODEL["checkpoint_filename"], **kwargs)
    vocab = hf_hub_download(filename=MODEL["vocab_filename"], **kwargs)
    for path, expected in ((checkpoint, MODEL["checkpoint_sha256"]), (vocab, MODEL["vocab_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
    model = load_checkpoint(checkpoint, torch.device(device))
    model.eval().requires_grad_(False)
    return model, ChatTokenizer(build_tokenizer(vocab, style="it"))
