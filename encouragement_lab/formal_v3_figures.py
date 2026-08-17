"""Publication figures from a validated formal-v3 analysis bundle.

This module is deliberately read-only: it validates the analysis artifact
hashes before deriving any display-only summaries from the tidy CSV files.
"""

from __future__ import annotations

import base64
import csv
from html import escape
import json
from math import sqrt
import os
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Mapping, Sequence

_cache_root = Path(gettempdir()) / "formal-v3-matplotlib-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "formal-v3"
import matplotlib.pyplot as plt  # noqa: E402

from .formal_v3_analysis import AXES
from .formal_v3_records import ARMS
from .personas import PERSONA_QUADRANTS, expected_template_ids
from .records import file_checksum


AXIS_LABELS = {"joyful": "joyful", "grief_stricken": "grief-stricken", "furious": "furious"}
ARM_LABELS = {"feedback_only": "Feedback only", "neutral": "Neutral filler", "supportive": "Supportive filler"}
COLORS = {"feedback_only": "#4D4D4D", "neutral": "#4477AA", "supportive": "#EE6677"}
MARKERS = {"feedback_only": "o", "neutral": "s", "supportive": "^"}
LINESTYLES = {"feedback_only": "-", "neutral": "--", "supportive": "-."}
REQUIRED_ARTIFACTS = frozenset({"rounds.csv", "seed_contrasts.csv", "co_primary.csv", "moderation.csv"})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_bundle(source: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Load only a complete, hash-validated v3 analysis bundle."""

    summary_path = source / "summary.json"
    if not summary_path.is_file():
        raise ValueError("analysis bundle has no summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1 or not isinstance(summary.get("artifact_sha256"), dict):
        raise ValueError("unsupported or incomplete formal-v3 analysis summary")
    hashes: Mapping[str, str] = summary["artifact_sha256"]
    if not REQUIRED_ARTIFACTS.issubset(hashes):
        raise ValueError("formal-v3 analysis bundle is missing required figure artifacts")
    for name, expected in hashes.items():
        path = source / name
        if not path.is_file() or not isinstance(expected, str) or file_checksum(path) != expected:
            raise ValueError(f"analysis artifact hash mismatch: {name}")
    if tuple(summary.get("expected_seeds", ())) != tuple(range(3001, 3011)):
        raise ValueError("unexpected formal-v3 seed schedule")
    inference = summary.get("inference")
    if not isinstance(inference, Mapping) or len(inference.get("co_primary", ())) != len(AXES) or len(inference.get("moderation", ())) != len(AXES) * 3:
        raise ValueError("formal-v3 inference summary has unexpected dimensions")
    rows = {name: _read_csv(source / name) for name in REQUIRED_ARTIFACTS}
    return summary, rows


def _complete_trajectory(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, str]], axis: str
) -> tuple[list[int], dict[str, tuple[list[float], list[float]]]]:
    values = {(int(r["seed"]), r["arm"], r["quadrant"], r["template"], int(r["round"])): float(r["projection"]) for r in rows if r["axis"] == axis}
    complete: list[int] = []
    for seed in summary["expected_seeds"]:
        required = ((seed, arm, quadrant, template, round_number) for arm in ARMS for quadrant in PERSONA_QUADRANTS for template in expected_template_ids(quadrant) for round_number in range(1, 6))
        if all(key in values for key in required):
            complete.append(seed)
    if not complete:
        raise ValueError(f"no complete all-arm seed block for {axis}")
    curves: dict[str, list[list[float]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in complete:
            curve = [mean(mean(values[(seed, arm, quadrant, template, round_number)] for template in expected_template_ids(quadrant)) for quadrant in PERSONA_QUADRANTS) for round_number in range(1, 6)]
            curves[arm].append([value - curve[0] for value in curve])
    return complete, {arm: ([mean(c[i] for c in curves[arm]) for i in range(5)], [stdev(c[i] for c in curves[arm]) / sqrt(len(curves[arm])) if len(curves[arm]) > 1 else 0.0 for i in range(5)]) for arm in ARMS}


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "savefig.dpi": 300})


def _save(fig: Any, stem: Path) -> None:
    for suffix, metadata in ((".svg", {"Creator": "formal-v3 figure pipeline", "Date": None}), (".pdf", {"Creator": "formal-v3 figure pipeline", "CreationDate": None, "ModDate": None}), (".png", {"Software": "formal-v3 figure pipeline"})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def _plot_trajectories(output: Path, summary: Mapping[str, Any], rounds: Sequence[Mapping[str, str]]) -> list[int]:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), sharex=True, constrained_layout=True)
    complete_all: list[int] | None = None
    for axis, key in zip(axes, AXES, strict=True):
        complete, trajectory = _complete_trajectory(summary, rounds, key)
        complete_all = complete if complete_all is None else [seed for seed in complete_all if seed in complete]
        for arm in ARMS:
            center, error = trajectory[arm]
            axis.plot(range(1, 6), center, label=ARM_LABELS[arm], color=COLORS[arm], marker=MARKERS[arm], linestyle=LINESTYLES[arm], linewidth=1.6, markersize=4)
            axis.fill_between(range(1, 6), [x-y for x, y in zip(center, error)], [x+y for x, y in zip(center, error)], color=COLORS[arm], alpha=.14, linewidth=0)
        axis.axhline(0, color="#999999", linewidth=.7)
        axis.set_title(AXIS_LABELS[key])
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("Consecutive unsuccessful guess")
        axis.set_ylabel("Change from R1 in\ncosine projection alignment")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Three-arm five-round trajectories", y=1.02)
    _save(fig, output / "formal-v3-trajectories")
    return complete_all or []


def _p_text(value: str) -> str:
    numeric = float(value)
    return "p_adj<.001" if numeric < .001 else f"p_adj={numeric:.3f}"


def _plot_primary(output: Path, rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    co_primary = {row["axis"]: row for row in rows["co_primary.csv"]}
    seed_points = {axis: [] for axis in AXES}
    for row in rows["seed_contrasts.csv"]:
        if row["contrast"] == "supportive_minus_neutral":
            seed_points[row["axis"]].append((int(row["seed"]), float(row["overall_slope"])))
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), sharey=True, constrained_layout=True)
    for axis, key in zip(axes, AXES, strict=True):
        points = sorted(seed_points[key])
        row = co_primary[key]
        axis.scatter(range(len(points)), [value for _, value in points], color="#999999", s=24, zorder=2, label="Seed effect")
        axis.errorbar((len(points)-1)/2, float(row["mean"]), yerr=float(row["standard_error"]), fmt="D", color="#332288", capsize=4, markersize=6, zorder=3, label="Mean ± SE")
        axis.axhline(0, color="#999999", linewidth=.7)
        axis.set_xticks(range(len(points)), [str(seed) for seed, _ in points], rotation=45, ha="right", fontsize=7)
        axis.set_title(AXIS_LABELS[key])
        axis.set_xlabel("Seed")
        axis.annotate(_p_text(row["holm_adjusted_p"]), (.02, .96), xycoords="axes fraction", va="top", fontsize=8)
    axes[0].set_ylabel("Supportive − neutral\nR1–R5 OLS slope difference")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Planned supportive − neutral slope effects", y=1.02)
    _save(fig, output / "formal-v3-planned-slope-effects")


def _plot_moderation(output: Path, rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    moderation = {(row["axis"], row["effect"]): row for row in rows["moderation.csv"]}
    effects = ("extraversion", "neuroticism", "interaction")
    labels = ("Extraversion", "Neuroticism", "E × N")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), sharey=True, constrained_layout=True)
    for axis, key in zip(axes, AXES, strict=True):
        for index, effect in enumerate(effects):
            row = moderation[(key, effect)]
            axis.errorbar(float(row["mean"]), index, xerr=float(row["standard_error"]), fmt="o", color=("#117733", "#CC6677", "#AA4499")[index], capsize=3)
            axis.annotate(_p_text(row["holm_adjusted_p"]), (float(row["mean"]), index), xytext=(5, 5), textcoords="offset points", fontsize=7)
        axis.axvline(0, color="#999999", linewidth=.7)
        axis.set_yticks(range(3), labels)
        axis.invert_yaxis()
        axis.set_title(AXIS_LABELS[key])
        axis.set_xlabel("Persona moderation of\nslope contrast")
        axis.margins(x=.3)
        axis.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3), useMathText=True)
    fig.suptitle("Planned persona moderation: supportive − neutral", y=1.02)
    _save(fig, output / "formal-v3-persona-moderation")


def _report(output: Path, summary: Mapping[str, Any], complete: Sequence[int]) -> None:
    images = "".join(f'<img alt="{name}" src="data:image/svg+xml;base64,{base64.b64encode((output / name).read_bytes()).decode("ascii")}">' for name in ("formal-v3-trajectories.svg", "formal-v3-planned-slope-effects.svg", "formal-v3-persona-moderation.svg"))
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>formal-v3 figures</title><style>body{{font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#222}}img{{width:100%;height:auto;margin:12px 0 28px}}.note{{color:#555}}</style></head><body><h1>formal-v3 discrete-emotion trajectory figures</h1><p class="note">Complete all-arm seed blocks: {len(complete)}. Values are layer-17 cosine projection alignment to external discrete-emotion vectors; they are prompt/context-sensitive representations, not subjective experience.</p>{images}</body></html>'''
    (output / "formal-v3-report.html").write_text(html, encoding="utf-8")


def write_figure_bundle(source: Path, destination: Path) -> dict[str, Any]:
    """Create a non-overwriting SVG/PDF/PNG/HTML bundle from validated v3 data."""

    if destination.exists():
        raise FileExistsError(f"figure destination already exists: {destination}")
    summary, rows = _load_bundle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="formal-v3-figures-", dir=destination.parent) as raw:
        staging = Path(raw)
        _style()
        complete = _plot_trajectories(staging, summary, rows["rounds.csv"])
        _plot_primary(staging, rows)
        _plot_moderation(staging, rows)
        _report(staging, summary, complete)
        hashes = {path.name: file_checksum(path) for path in sorted(staging.iterdir()) if path.is_file()}
        manifest = {"schema_version": 1, "analysis_summary_sha256": file_checksum(source / "summary.json"), "complete_all_arm_seeds": complete, "artifacts_sha256": hashes}
        (staging / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(destination)
    return manifest
