from __future__ import annotations

import copy
import json
from pathlib import Path

import run_formal_v3_freeze_audit as audit_module


ROOT = Path(__file__).resolve().parents[1]


def _audit_without_rehashing_model() -> dict:
    return audit_module.audit(
        model_checksum=lambda _path: audit_module.FROZEN_MODEL_SHA256,
    )


def test_candidate_freeze_audit_accepts_checked_in_inputs() -> None:
    manifest = _audit_without_rehashing_model()

    assert manifest["passed"] is True
    assert manifest["formal_v3_freeze_id"] == "formal-v3-2026-08-17"
    assert manifest["collection_started"] is True
    assert manifest["runner_source_snapshot_sha256"] == audit_module.source_snapshot_checksum()
    assert all(check["passed"] for check in manifest["checks"])


def test_candidate_freeze_audit_rejects_nonfinite_smoke_readout(tmp_path) -> None:
    rows = [json.loads(line) for line in audit_module.DEFAULT_SMOKE.read_text().splitlines()]
    altered = copy.deepcopy(rows)
    altered[0]["trajectory"][0]["readout"]["joyful"]["layers"]["17"] = float("nan")
    smoke = tmp_path / "smoke.jsonl"
    smoke.write_text("\n".join(json.dumps(row) for row in altered) + "\n", encoding="utf-8")

    manifest = audit_module.audit(
        smoke=smoke,
        model_checksum=lambda _path: audit_module.FROZEN_MODEL_SHA256,
    )

    smoke_check = next(check for check in manifest["checks"] if check["name"] == "real_backend_smoke")
    assert manifest["passed"] is False
    assert smoke_check["passed"] is False


def test_main_refuses_to_overwrite_manifest(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "existing.json"
    destination.write_text("already here\n", encoding="utf-8")
    monkeypatch.setattr(audit_module, "parse_args", lambda: type("Args", (), {
        "output": destination,
        "protocol": audit_module.DEFAULT_PROTOCOL,
        "prompts": audit_module.DEFAULT_PROMPTS,
        "directions": audit_module.DEFAULT_DIRECTIONS,
        "model": audit_module.DEFAULT_MODEL,
        "smoke": audit_module.DEFAULT_SMOKE,
    })())
    try:
        audit_module.main()
    except SystemExit as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("existing manifest was accepted")
