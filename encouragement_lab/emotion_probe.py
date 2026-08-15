"""Read-only internal activation directions for positive and negative affect."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np


# Fixed after the pre-data Qwen pilot: a contiguous five-layer late-model band.
# Wider bands reached into middle layers where direction signs did not transfer
# to held-out wording. Formal runs must not tune this range post hoc.
DEFAULT_LAYERS = tuple(range(-5, -10, -1))
AXES = ("positive", "negative", "frustration")
BROAD_AFFECT_AXES = ("positive", "negative")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


@dataclass(frozen=True)
class EmotionMaterials:
    train_templates: tuple[str, ...]
    heldout_templates: tuple[str, ...]
    train_target_words: Mapping[str, tuple[str, ...]]
    heldout_target_words: Mapping[str, tuple[str, ...]]
    train_neutral_words: tuple[str, ...]
    heldout_neutral_words: tuple[str, ...]
    train_frustration_pairs: tuple[tuple[str, str], ...]
    heldout_frustration_pairs: tuple[tuple[str, str], ...]

    @classmethod
    def from_prompts(cls, prompts: Mapping[str, str]) -> "EmotionMaterials":
        return cls(
            train_templates=tuple(_lines(prompts["emotion.train.templates"])),
            heldout_templates=tuple(_lines(prompts["emotion.heldout.templates"])),
            train_target_words={
                axis: tuple(_lines(prompts[f"emotion.train.{axis}_words"]))
                for axis in BROAD_AFFECT_AXES
            },
            heldout_target_words={
                axis: tuple(_lines(prompts[f"emotion.heldout.{axis}_words"]))
                for axis in BROAD_AFFECT_AXES
            },
            train_neutral_words=tuple(
                _lines(prompts["emotion.train.neutral_words"])
            ),
            heldout_neutral_words=tuple(
                _lines(prompts["emotion.heldout.neutral_words"])
            ),
            train_frustration_pairs=_matched_pairs(
                prompts["emotion.train.frustration_targets"],
                prompts["emotion.train.frustration_controls"],
                label="training frustration",
            ),
            heldout_frustration_pairs=_matched_pairs(
                prompts["emotion.heldout.frustration_targets"],
                prompts["emotion.heldout.frustration_controls"],
                label="held-out frustration",
            ),
        )

    def pairs(self, axis: str, *, heldout: bool) -> list[tuple[str, str]]:
        if axis not in AXES:
            raise ValueError(f"unknown emotion axis: {axis}")
        if axis == "frustration":
            return list(
                self.heldout_frustration_pairs
                if heldout
                else self.train_frustration_pairs
            )
        templates = self.heldout_templates if heldout else self.train_templates
        targets = (
            self.heldout_target_words[axis]
            if heldout
            else self.train_target_words[axis]
        )
        neutrals = self.heldout_neutral_words if heldout else self.train_neutral_words
        if not templates or not targets or not neutrals:
            raise ValueError(f"empty materials for {axis}")
        pairs: list[tuple[str, str]] = []
        for template in templates:
            for index, target in enumerate(targets):
                neutral = neutrals[index % len(neutrals)]
                pairs.append(
                    (
                        template.format(emotion=target),
                        template.format(emotion=neutral),
                    )
                )
        return pairs

    def checksum(self) -> str:
        payload = {
            "train_templates": self.train_templates,
            "heldout_templates": self.heldout_templates,
            "train_target_words": self.train_target_words,
            "heldout_target_words": self.heldout_target_words,
            "train_neutral_words": self.train_neutral_words,
            "heldout_neutral_words": self.heldout_neutral_words,
            "train_frustration_pairs": self.train_frustration_pairs,
            "heldout_frustration_pairs": self.heldout_frustration_pairs,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _matched_pairs(
    target_text: str, control_text: str, *, label: str
) -> tuple[tuple[str, str], ...]:
    targets = _lines(target_text)
    controls = _lines(control_text)
    if not targets or len(targets) != len(controls):
        raise ValueError(
            f"{label} materials must contain equal non-zero target/control lines"
        )
    return tuple(zip(targets, controls, strict=True))


@dataclass
class EmotionDirections:
    model_type: str
    model_checksum: str | None
    materials_checksum: str
    directions: dict[str, dict[int, np.ndarray]]
    validation: dict[str, Any]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for axis, layers in self.directions.items():
            for layer, direction in layers.items():
                arrays[f"{axis}__{layer}"] = direction.astype(np.float32)
        arrays["__metadata__"] = np.asarray(
            json.dumps(
                {
                    "model_type": self.model_type,
                    "model_checksum": self.model_checksum,
                    "materials_checksum": self.materials_checksum,
                    "validation": self.validation,
                },
                ensure_ascii=False,
            )
        )
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "EmotionDirections":
        data = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(data["__metadata__"]))
        directions: dict[str, dict[int, np.ndarray]] = {axis: {} for axis in AXES}
        for key in data.files:
            if key == "__metadata__":
                continue
            axis, layer = key.split("__", 1)
            directions[axis][int(layer)] = data[key].astype(np.float32)
        return cls(
            model_type=metadata["model_type"],
            model_checksum=metadata.get("model_checksum"),
            materials_checksum=metadata["materials_checksum"],
            directions=directions,
            validation=metadata["validation"],
        )

    def assert_valid(
        self,
        materials: EmotionMaterials,
        *,
        model_checksum: str | None = None,
        minimum_accuracy: float = 0.75,
    ) -> None:
        if self.materials_checksum != materials.checksum():
            raise ValueError(
                "emotion-direction materials changed; retrain the artifact before use"
            )
        if model_checksum is not None and self.model_checksum != model_checksum:
            raise ValueError(
                "emotion directions were trained on a different model snapshot"
            )
        for axis in AXES:
            accuracy = self.validation.get(axis, {}).get("aggregate_accuracy")
            if not isinstance(accuracy, (int, float)) or accuracy < minimum_accuracy:
                raise ValueError(
                    f"emotion direction {axis!r} lacks held-out aggregate accuracy "
                    f">= {minimum_accuracy}"
                )


def _projection(hidden: np.ndarray, direction: np.ndarray) -> float:
    magnitude = float(np.linalg.norm(direction) * np.linalg.norm(hidden))
    if not magnitude:
        raise ValueError("emotion direction has zero magnitude")
    return float(hidden @ direction / magnitude)


class EmotionProbe:
    """Train and score repeng directions without ever steering generation."""

    def __init__(self, model: Any, tokenizer: Any, directions: EmotionDirections):
        if directions.model_type != model.config.model_type:
            raise ValueError(
                f"direction model_type {directions.model_type!r} does not match "
                f"model {model.config.model_type!r}"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.directions = directions
        missing = [axis for axis in AXES if not directions.directions.get(axis)]
        if missing:
            raise ValueError("emotion directions missing axes: " + ", ".join(missing))

    @classmethod
    def train(
        cls,
        model: Any,
        tokenizer: Any,
        materials: EmotionMaterials,
        render_user_prompt: Callable[[str], str],
        *,
        layers: Sequence[int] = DEFAULT_LAYERS,
        batch_size: int = 2,
        model_checksum: str | None = None,
        minimum_accuracy: float = 0.75,
    ) -> "EmotionProbe":
        try:
            from repeng import ControlVector, DatasetEntry
        except ImportError as error:  # pragma: no cover - real environment only
            raise RuntimeError("repeng is required to train emotion directions") from error

        directions: dict[str, dict[int, np.ndarray]] = {}
        for axis in AXES:
            dataset = [
                DatasetEntry(
                    positive=render_user_prompt(target),
                    negative=render_user_prompt(neutral),
                )
                for target, neutral in materials.pairs(axis, heldout=False)
            ]
            vector = ControlVector.train(
                model,
                tokenizer,
                dataset,
                hidden_layers=list(layers),
                batch_size=batch_size,
                method="pca_diff",
            )
            directions[axis] = vector.directions

        provisional = EmotionDirections(
            model_type=model.config.model_type,
            model_checksum=model_checksum,
            materials_checksum=materials.checksum(),
            directions=directions,
            validation={},
        )
        probe = cls(model, tokenizer, provisional)
        validation = probe.validate(materials, render_user_prompt)
        failed = [
            axis
            for axis, result in validation.items()
            if result["aggregate_accuracy"] < minimum_accuracy
        ]
        if failed:
            raise ValueError(
                "held-out emotion separation failed for: " + ", ".join(failed)
            )
        probe.directions.validation = validation
        return probe

    def _hidden_states(self, texts: Sequence[str]) -> dict[int, np.ndarray]:
        from repeng.extract import batched_get_hiddens

        layers = sorted(
            {layer for axis in AXES for layer in self.directions.directions[axis]}
        )
        return batched_get_hiddens(
            self.model, self.tokenizer, list(texts), layers, batch_size=2
        )

    def score_text(self, rendered_text: str) -> dict[str, Any]:
        hidden = self._hidden_states([rendered_text])
        result: dict[str, Any] = {}
        for axis in AXES:
            by_layer = {
                str(layer): _projection(hidden[layer][0], direction)
                for layer, direction in self.directions.directions[axis].items()
            }
            result[axis] = {
                "layers": by_layer,
                "median": float(median(by_layer.values())),
            }
        return result

    def validate(
        self,
        materials: EmotionMaterials,
        render_user_prompt: Callable[[str], str],
    ) -> dict[str, Any]:
        validation: dict[str, Any] = {}
        for axis in AXES:
            pairs = materials.pairs(axis, heldout=True)
            texts = [
                render_user_prompt(text)
                for pair in pairs
                for text in pair
            ]
            hidden = self._hidden_states(texts)
            layer_results: dict[str, Any] = {}
            projections_by_layer: dict[int, list[float]] = {}
            for layer, direction in self.directions.directions[axis].items():
                projections = [
                    _projection(row, direction) for row in hidden[layer]
                ]
                projections_by_layer[layer] = projections
                margins = [
                    projections[index] - projections[index + 1]
                    for index in range(0, len(projections), 2)
                ]
                layer_results[str(layer)] = {
                    "accuracy": sum(value > 0 for value in margins) / len(margins),
                    "median_margin": float(median(margins)),
                }
            aggregate_margins = []
            for index in range(0, len(texts), 2):
                target_score = median(
                    values[index] for values in projections_by_layer.values()
                )
                neutral_score = median(
                    values[index + 1] for values in projections_by_layer.values()
                )
                aggregate_margins.append(target_score - neutral_score)
            validation[axis] = {
                "median_layer_accuracy": float(
                    median(item["accuracy"] for item in layer_results.values())
                ),
                "aggregate_accuracy": sum(
                    value > 0 for value in aggregate_margins
                )
                / len(aggregate_margins),
                "aggregate_median_margin": float(median(aggregate_margins)),
                "layers": layer_results,
            }
        return validation
