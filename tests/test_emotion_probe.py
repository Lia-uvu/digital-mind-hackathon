import numpy as np
import pytest

from encouragement_lab.emotion_probe import (
    EmotionDirections,
    EmotionMaterials,
    EmotionProbe,
)
from encouragement_lab.prompt_loader import load_prompts


def test_emotion_materials_are_frozen_and_task_independent(tmp_path) -> None:
    prompts = load_prompts("prompts.md")
    materials = EmotionMaterials.from_prompts(prompts)
    pairs = {
        (axis, heldout): materials.pairs(axis, heldout=heldout)
        for axis in ("positive", "negative", "frustration")
        for heldout in (False, True)
    }

    assert all(len(pairs[(axis, heldout)]) == 12 for axis in ("positive", "negative") for heldout in (False, True))
    assert len(pairs[("frustration", False)]) == 12
    assert len(pairs[("frustration", True)]) == 12
    assert len(materials.train_frustration_pairs) == len(materials.heldout_frustration_pairs) == 12
    assert all(target != control for pair in pairs.values() for target, control in pair)

    frustration_text = " ".join(
        text for heldout in (False, True) for pair in pairs[("frustration", heldout)] for text in pair
    ).lower()
    assert "frustrat" not in frustration_text
    assert "mastermind" not in frustration_text
    assert "encourag" not in frustration_text

    combined = " ".join(text for pair_list in pairs.values() for pair in pair_list for text in pair).lower()
    assert "mastermind" not in combined
    assert "encourag" not in combined
    assert "failure" not in combined


def test_emotion_directions_round_trip_without_pickle(tmp_path) -> None:
    path = tmp_path / "directions.npz"
    original = EmotionDirections(
        model_type="qwen2",
        model_checksum="b" * 64,
        materials_checksum="a" * 64,
        directions={
            "positive": {11: np.asarray([1.0, 0.0], dtype=np.float32)},
            "negative": {11: np.asarray([0.0, 1.0], dtype=np.float32)},
            "frustration": {11: np.asarray([-1.0, 0.0], dtype=np.float32)},
        },
        validation={
            "positive": {"aggregate_accuracy": 1.0},
            "negative": {"aggregate_accuracy": 1.0},
            "frustration": {"aggregate_accuracy": 1.0},
        },
    )
    original.save(path)
    loaded = EmotionDirections.load(path)

    assert loaded.model_type == "qwen2"
    assert loaded.validation == original.validation
    np.testing.assert_array_equal(
        loaded.directions["positive"][11], original.directions["positive"][11]
    )
    np.testing.assert_array_equal(
        loaded.directions["frustration"][11], original.directions["frustration"][11]
    )
    assert loaded.validation["frustration"] == {"aggregate_accuracy": 1.0}


def test_probe_rejects_directions_missing_frustration_axis() -> None:
    class FakeModel:
        class config:
            model_type = "qwen2"

    directions = EmotionDirections(
        model_type="qwen2",
        model_checksum=None,
        materials_checksum="a" * 64,
        directions={
            "positive": {11: np.asarray([1.0], dtype=np.float32)},
            "negative": {11: np.asarray([1.0], dtype=np.float32)},
        },
        validation={},
    )

    with pytest.raises(ValueError, match="missing axes: frustration"):
        EmotionProbe(FakeModel(), object(), directions)
