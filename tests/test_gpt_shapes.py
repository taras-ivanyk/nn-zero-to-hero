import math

import pytest
import torch

from gpt import GPT, CharTokenizer, GPTConfig


@pytest.fixture
def config():
    return GPTConfig(vocab_size=65, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return GPT(config).eval()


def test_forward_shapes_without_targets(model, config):
    idx = torch.randint(0, config.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, config.vocab_size)
    assert loss is None


def test_loss_is_finite_and_near_uniform_at_init(model, config):
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))
    targets = torch.randint(0, config.vocab_size, (4, config.block_size))
    logits, loss = model(idx, targets)
    assert logits.shape == (4 * config.block_size, config.vocab_size)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(math.log(config.vocab_size), rel=0.25)


def test_attention_is_causal(model, config):
    idx = torch.randint(0, config.vocab_size, (1, 12))
    changed = idx.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % config.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-6)
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_generate_extends_beyond_block_size(model, config):
    torch.manual_seed(0)
    out = model.generate(torch.zeros((2, 1), dtype=torch.long), max_new_tokens=config.block_size + 5)
    assert out.shape == (2, config.block_size + 6)
    assert out.max() < config.vocab_size


def test_one_training_step_reduces_loss(config):
    torch.manual_seed(0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (8, config.block_size))
    targets = torch.roll(idx, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, before = model(idx, targets)
    for _ in range(5):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, after = model(idx, targets)
    assert after.item() < before.item()


def test_char_tokenizer_round_trip():
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
