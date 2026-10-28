import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.config import preset  # noqa: E402
from viron.model import Viron  # noqa: E402
from viron.serve import completion, encode_chat, generate, make_server  # noqa: E402
from viron.tokenizer import ByteTokenizer  # noqa: E402

TOK = ByteTokenizer()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Viron(preset("nano", vocab_size=TOK.vocab_size)).eval()


def test_encode_chat_uses_reserved_tokens_in_order():
    ids = encode_chat(TOK, [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    t = TOK.token_id
    assert ids == [
        t("<|bos|>"),
        t("<|system|>"),
        *TOK.encode("s"),
        t("<|end|>"),
        t("<|user|>"),
        *TOK.encode("hi"),
        t("<|end|>"),
        t("<|assistant|>"),
    ]
    with pytest.raises(ValueError, match="unsupported role"):
        encode_chat(TOK, [{"role": "tool", "content": "x"}])


def test_generate_returns_one_logprob_per_token(model):
    ids, lps, finish = generate(
        model, encode_chat(TOK, [{"role": "user", "content": "hi"}]), 12, 0.0, 1.0, set(), "cpu"
    )
    assert finish == "length" and len(ids) == len(lps) == 12
    assert all(lp <= 0 for lp in lps)


def test_generate_stops_on_stop_token(model):
    prompt = encode_chat(TOK, [{"role": "user", "content": "hi"}])
    first, _, _ = generate(model, prompt, 1, 0.0, 1.0, set(), "cpu")
    ids, lps, finish = generate(model, prompt, 12, 0.0, 1.0, {first[0]}, "cpu")  # greedy: first token is a stop
    assert (ids, lps, finish) == ([], [], "stop")


def test_completion_is_openai_shaped_with_logprobs(model):
    res = completion(
        model,
        TOK,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0,
            "logprobs": True,
            "lora": [{"id": 0, "scale": 1}],
        },
        "cpu",
    )
    choice = res["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert len(choice["logprobs"]["content"]) == res["usage"]["completion_tokens"]
    assert {"token", "logprob"} <= set(choice["logprobs"]["content"][0])
    with pytest.raises(ValueError, match="non-empty"):
        completion(model, TOK, {"messages": []}, "cpu")


def test_http_server_end_to_end(model):
    srv = make_server(model, TOK, "127.0.0.1", 0, "cpu", name="viron-test")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(body: dict):
        req = urllib.request.Request(
            f"{base}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}
        )
        return json.load(urllib.request.urlopen(req, timeout=30))

    try:
        assert json.load(urllib.request.urlopen(f"{base}/health"))["model"] == "viron-test"
        res = post({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4, "logprobs": True})
        assert res["model"] == "viron-test" and res["usage"]["completion_tokens"] <= 4
        with pytest.raises(urllib.error.HTTPError) as err:
            post({"messages": "nope"})
        assert err.value.code == 400
    finally:
        srv.shutdown()
