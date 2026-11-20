import pytest

from tokenizer import BasicTokenizer, RegexTokenizer

TRAIN_TEXT = (
    "The quick brown fox jumps over the lazy dog. The dog sleeps; the fox runs! "
    "In 2024 the fox ran 12345 metres, and the dog didn't. "
) * 20

ROUND_TRIP_TEXTS = [
    "",
    "a",
    "hello world",
    "The fox's den isn't far: 3.14159 km.\n\tNext line.",
    "Ünïcödé, 日本語, emoji 🦊🐶 and  multiple   spaces",
]


@pytest.fixture(params=[BasicTokenizer, RegexTokenizer], ids=["basic", "regex"])
def trained(request):
    tok = request.param()
    tok.train(TRAIN_TEXT, vocab_size=300)
    return tok


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_encode_decode_round_trip(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_training_compresses(trained):
    ids = trained.encode(TRAIN_TEXT)
    assert len(ids) < len(TRAIN_TEXT.encode("utf-8"))
    assert len(trained.vocab) == 300


def test_vocab_entries_are_concatenated_pairs(trained):
    for (p0, p1), idx in trained.merges.items():
        assert trained.vocab[idx] == trained.vocab[p0] + trained.vocab[p1]


def test_untrained_tokenizer_is_raw_bytes():
    tok = BasicTokenizer()
    assert tok.encode("hi!") == list("hi!".encode("utf-8"))


def test_save_load_round_trip(trained, tmp_path):
    path = tmp_path / "tok.json"
    trained.save(path)
    reloaded = type(trained)().load(path)
    assert reloaded.merges == trained.merges
    assert reloaded.vocab == trained.vocab
    assert reloaded.encode(TRAIN_TEXT) == trained.encode(TRAIN_TEXT)


def test_load_rejects_other_tokenizer_type(tmp_path):
    tok = BasicTokenizer()
    tok.train(TRAIN_TEXT, vocab_size=260)
    path = tmp_path / "basic.json"
    tok.save(path)
    with pytest.raises(ValueError):
        RegexTokenizer().load(path)


def test_gpt4_split_pattern():
    chunks = RegexTokenizer().compiled_pattern.findall("Hello world! It's 2024   ok\n")
    assert chunks == ["Hello", " world", "!", " It", "'s", " ", "202", "4", "  ", " ok", "\n"]


def test_regex_merges_stay_inside_chunks():
    tok = RegexTokenizer()
    tok.train(TRAIN_TEXT, vocab_size=300)
    for idx in range(256, 300):
        token = tok.vocab[idx]
        # outside pure-whitespace chunks, a space can only be the first byte of a chunk
        if token.strip():
            assert b" " not in token[1:]


def test_basic_merges_can_cross_word_boundaries():
    tok = BasicTokenizer()
    tok.train(TRAIN_TEXT, vocab_size=300)
    assert any(b" " in tok.vocab[idx][1:] for idx in range(256, 300))
