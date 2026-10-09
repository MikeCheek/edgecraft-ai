import asyncio
import json
import math

import pytest
from aiohttp import web

from app.services import llm_client
from app.services.training_review import analyze_training, review_training, validate_changes

LABELS = ["basil", "mint_plant", "oregano", "parsley", "rosemary", "sage", "thyme"]
COUNTS = {"mint_plant": 41, "rosemary": 41, "parsley": 40, "sage": 29, "oregano": 27, "basil": 23, "thyme": 22}


def herb_run():
    """Mirrors a real run: MobileNetV3Small, 7 small classes, frozen warm-up
    for 16 epochs then fine-tuning, overfits, best val at epoch 39, early
    stopping off."""
    ms = []
    for e in range(1, 51):
        head = e <= 16
        train_acc = min(0.97, 0.15 + e * 0.04) if head else min(0.97, 0.75 + (e - 16) * 0.02)
        val_acc = 0.563 if e == 39 else min(0.5, 0.12 + e * 0.02) - (0.06 if e > 40 else 0)
        val_loss = 1.49 + abs(e - 21) * (0.02 if e < 21 else 0.023)
        ms.append({
            "epoch": e, "phase": "head" if head else "fine-tune",
            "accuracy": train_acc, "val_accuracy": val_acc, "loss": max(0.09, 2.0 - e * 0.04), "val_loss": val_loss,
            "val_f1": val_acc - 0.01, "val_precision": val_acc + 0.01, "val_recall": val_acc - 0.005,
            "val_confidence": 0.81, "val_ece": 0.374 if e == 50 else 0.2,
            "learning_rate": 1e-3 if head else 1e-4, "update_ratio": 0.01,
        })
    ms[-1].update(accuracy=0.969, val_accuracy=0.438)
    return {
        "id": "herbs", "task": "IMAGE_CLASSIFICATION", "dataset_id": None, "status": "completed",
        "base_model": "MobileNetV3Small", "input_shape": [128, 128, 3], "total_epochs": 50, "batch_size": 8,
        "learning_rate": 0.001, "dropout_rate": 0.3, "l2_reg": 0.0, "early_stopping": False,
        "freeze_encoder_epochs": 16, "trainable_layers": 0, "class_weighting": False,
        "augmentation": {"horizontal_flip": False, "random_rotation": 0},
        "metrics": ms, "labels": LABELS, "started_at": 0, "completed_at": 189,
        "run_info": {"num_train": 223, "num_val": 64, "num_classes": 7, "labels": LABELS,
                     "params_total": 943_200, "train_class_counts": COUNTS},
        "evaluation": {"split": "test", "num_samples": 30, "accuracy": 0.567, "macro_f1": 0.553, "labels": LABELS,
                       "per_class": [{"label": lbl, "precision": .5, "recall": .5, "f1": f, "support": 4}
                                     for lbl, f in zip(LABELS, [.6, .29, .75, .29, .67, .62, .67])],
                       "confusion_matrix": [[3, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 0], [0, 0, 3, 1, 0, 0, 0],
                                            [3, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 3, 2, 0], [0, 1, 0, 0, 0, 4, 0],
                                            [0, 0, 0, 0, 1, 1, 2]]},
    }


def test_analysis_finds_what_matters_and_is_deterministic():
    session = herb_run()
    a = analyze_training(session, board="ESP32_S3_N16R8")
    b = analyze_training(session, board="ESP32_S3_N16R8")
    assert a["score"] == b["score"] and a["findings"] == b["findings"]
    assert 15 <= a["score"] <= 50 and a["score_label"] == "poor"

    titles = " | ".join(f["title"] for f in a["findings"])
    assert "overfitting" in titles
    assert "worse than the best epoch" in titles
    assert "Too few training samples" in titles
    assert "Test result is noisy" in titles
    evidence = " ".join(f["evidence"] for f in a["findings"])
    assert "epoch 39" in evidence and "56.3%" in evidence and "43.8%" in evidence
    assert "parsley -> basil (3x)" in evidence

    recs = {r["title"]: r for r in a["rule_suggestions"]}
    es = next(r for t, r in recs.items() if "early stopping" in t)
    assert es["changes"]["early_stopping"] is True and es["priority"] == "high"
    reg = next(r for t, r in recs.items() if "Regularise" in t)
    assert reg["changes"]["dropout_rate"] == 0.5 and reg["changes"]["augmentation"]["horizontal_flip"] is True
    assert any("class weighting" in t for t in recs)
    # No recommendation re-proposes a value the run already used.
    for r in a["rule_suggestions"]:
        for k, v in r["changes"].items():
            assert a["facts"]["config"].get(k) != v
    assert a["facts"]["deployment"]["int8_flash_usage_pct"] < 10
    json.dumps(a)  # serialisable


def test_validate_changes_drops_noops_and_invalid_values():
    cur = {"learning_rate": 0.001, "batch_size": 8, "dropout_rate": 0.3, "augmentation": {"horizontal_flip": True},
           "base_model": "MobileNetV3Small", "early_stopping": False}
    changes, notes = validate_changes("IMAGE_CLASSIFICATION", {
        "learning_rate": "0.001", "batch_size": 8, "dropout_rate": 2, "base_model": "GPT-4 Vision",
        "augmentation": {"horizontal_flip": True, "random_rotation": 0.15, "bogus": 1}, "early_stopping": "true",
        "magic": 3,
    }, cur)
    assert changes == {"dropout_rate": 0.9, "augmentation": {"random_rotation": 0.15}, "early_stopping": True}
    assert any("GPT-4" in n for n in notes) and any("magic" in n for n in notes)
    assert validate_changes("KEYWORD_SPOTTING", {"input_shape": [40, 49, 1]}, {})[0] == {}


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here is the JSON:\n{"a": 1,}\nHope it helps.',
    '<think>{"draft": true}</think>{"a": 1}',
])
def test_extract_json_tolerates_wrappers(text):
    assert llm_client.extract_json(text) == {"a": 1}


def _run_with_fake_openrouter(monkeypatch, handler, coro_factory, timeout="5"):
    async def main():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(llm_client, "OPENROUTER_URL", f"http://127.0.0.1:{port}/v1/chat/completions")
        try:
            return await coro_factory()
        finally:
            await runner.cleanup()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", timeout)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    return asyncio.run(main())


_real_sleep = asyncio.sleep


async def _fast_sleep(delay, *a, **k):
    await _real_sleep(min(delay, 0.01))


def _reply(content):
    return web.json_response({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


def test_chat_json_retries_rate_limit_timeout_and_json_mode(monkeypatch):
    calls = []

    async def handler(request):
        body = await request.json()
        calls.append(body)
        n = len(calls)
        if n == 1:
            return web.Response(status=429, text="rate limited", headers={"Retry-After": "0"})
        if n == 2:
            await _real_sleep(1.5)  # exceeds the 1 s per-attempt timeout
            return _reply("{}")
        if "response_format" in body:
            return web.Response(status=400, text='{"error": "response_format json_object is not supported"}')
        return _reply('Here you go: {"ok": true}')

    out = _run_with_fake_openrouter(
        monkeypatch, handler,
        lambda: llm_client.chat_json([{"role": "user", "content": "hi"}], "openrouter", "m", max_attempts=4),
        timeout="1")
    assert out == {"ok": True}
    assert len(calls) == 4 and "response_format" not in calls[-1]


def test_review_falls_back_to_rules_when_ai_fails(monkeypatch):
    async def handler(request):
        return web.Response(status=503, text="overloaded")

    review = _run_with_fake_openrouter(
        monkeypatch, handler, lambda: review_training(herb_run(), "ESP32_S3_N16R8", "openrouter", "m"))
    assert review["suggestions_source"] == "rules" and review["suggestions"]
    assert "503" in review["ai_error"] and review["score"] == analyze_training(herb_run(), "ESP32_S3_N16R8")["score"]


def test_review_validates_ai_suggestions(monkeypatch):
    reply = {"summary": "Overfits a tiny dataset.", "suggestions": [
        {"title": "Lower LR", "priority": "high", "reasoning": "x", "changes": {"learning_rate": 0.001}},  # no-op
        {"title": "Stop early", "priority": "urgent", "reasoning": "best at 39",
         "changes": {"early_stopping": True, "epochs": 60, "ram_mb": 5}, "expected_effect": "keep 56%"},
        {"title": "Collect more thyme images", "category": "data", "reasoning": "22 samples", "changes": {}},
    ]}

    async def handler(request):
        return _reply("```json\n" + json.dumps(reply) + "\n```")

    review = _run_with_fake_openrouter(
        monkeypatch, handler, lambda: review_training(herb_run(), "ESP32_S3_N16R8", "openrouter", "m"))
    assert review["suggestions_source"] == "ai" and review["ai_summary"] == "Overfits a tiny dataset."
    titles = [s["title"] for s in review["suggestions"]]
    assert titles == ["Stop early", "Collect more thyme images"]
    stop = review["suggestions"][0]
    assert stop["changes"] == {"early_stopping": True, "epochs": 60} and stop["priority"] == "medium"
    assert any("ram_mb" in n for n in stop["rejected_changes"])
    assert not math.isnan(review["score"])
