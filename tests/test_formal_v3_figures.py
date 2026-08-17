from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from encouragement_lab.formal_v3_figures import write_figure_bundle
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "formal-v3-analysis"


def test_writes_complete_non_overwriting_figure_bundle(tmp_path: Path) -> None:
    output = tmp_path / "figures"
    manifest = write_figure_bundle(ANALYSIS, output)
    expected = {
        f"{stem}.{suffix}"
        for stem in ("formal-v3-trajectories", "formal-v3-planned-slope-effects", "formal-v3-persona-moderation")
        for suffix in ("svg", "pdf", "png")
    } | {"formal-v3-report.html"}
    assert expected == set(manifest["artifacts_sha256"])
    assert manifest["complete_all_arm_seeds"] == list(range(3001, 3011))
    for name, digest in manifest["artifacts_sha256"].items():
        assert file_checksum(output / name) == digest
    assert "subjective experience" in (output / "formal-v3-report.html").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_figure_bundle(ANALYSIS, output)


def test_rejects_tampered_analysis_artifact(tmp_path: Path) -> None:
    copied = tmp_path / "analysis"
    shutil.copytree(ANALYSIS, copied)
    (copied / "rounds.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_figure_bundle(copied, tmp_path / "figures")
