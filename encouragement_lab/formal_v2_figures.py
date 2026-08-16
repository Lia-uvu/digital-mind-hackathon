"""Publication figures from a validated formal-v2 analysis bundle."""

from __future__ import annotations

import base64
import csv
from html import escape
import json
from math import sqrt
import os
from pathlib import Path
from statistics import mean, stdev
from tempfile import gettempdir, TemporaryDirectory
from typing import Any, Mapping, Sequence

_cache_root = Path(gettempdir()) / "formal-v2-matplotlib-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "formal-v2"
import matplotlib.pyplot as plt  # noqa: E402

from .formal_v2_analysis import DERIVED_CONTRAST, PRIMARY_CONTRASTS
from .personas import PERSONA_QUADRANTS, expected_template_ids
from .records import file_checksum


ARMS = ("feedback_only", "neutral", "supportive")
ARM_LABELS = {
    "feedback_only": "Feedback only",
    "neutral": "Neutral filler",
    "supportive": "Supportive filler",
}
COLORS = {
    "feedback_only": "#4D4D4D",
    "neutral": "#4477AA",
    "supportive": "#EE6677",
}
MARKERS = {"feedback_only": "o", "neutral": "s", "supportive": "^"}
LINESTYLES = {"feedback_only": "-", "neutral": "--", "supportive": "-."}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_bundle(source: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for name, expected in summary["artifact_sha256"].items():
        path = source / name
        if not path.is_file() or file_checksum(path) != expected:
            raise ValueError(f"analysis artifact hash mismatch: {name}")
    return summary, _read_csv(source / "rounds.csv")


def _trajectory(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, str]]
) -> tuple[list[int], dict[str, dict[str, list[float] | int]]]:
    values = {
        (
            int(row["seed"]),
            row["arm"],
            row["quadrant"],
            row["template"],
            int(row["round"]),
        ): float(row["frustration_median"])
        for row in rows
    }
    complete: list[int] = []
    for seed in summary["expected_seeds"]:
        required = (
            (seed, arm, quadrant, template, round_number)
            for arm in ARMS
            for quadrant in PERSONA_QUADRANTS
            for template in expected_template_ids(quadrant)
            for round_number in range(1, 6)
        )
        if all(key in values for key in required):
            complete.append(seed)
    if not complete:
        raise ValueError("no complete all-arm seed block is available for figures")

    result: dict[str, dict[str, list[float] | int]] = {}
    for arm in ARMS:
        seed_curves: list[list[float]] = []
        for seed in complete:
            curve = []
            for round_number in range(1, 6):
                quadrant_means = [
                    mean(
                        values[(seed, arm, quadrant, template, round_number)]
                        for template in expected_template_ids(quadrant)
                    )
                    for quadrant in PERSONA_QUADRANTS
                ]
                curve.append(mean(quadrant_means))
            seed_curves.append([value - curve[0] for value in curve])
        means = [mean(curve[index] for curve in seed_curves) for index in range(5)]
        errors = [
            stdev(curve[index] for curve in seed_curves) / sqrt(len(seed_curves))
            if len(seed_curves) > 1
            else 0.0
            for index in range(5)
        ]
        result[arm] = {"mean": means, "se": errors, "n": len(seed_curves)}
    return complete, result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def _save(fig: Any, stem: Path) -> None:
    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Creator": "formal-v2 figure pipeline", "Date": None},
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "formal-v2 figure pipeline",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        stem.with_suffix(".png"),
        bbox_inches="tight",
        metadata={"Software": "formal-v2 figure pipeline"},
    )
    plt.close(fig)


def _plot_trajectory(
    output: Path, complete: Sequence[int], trajectory: Mapping[str, Any]
) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    rounds = [1, 2, 3, 4, 5]
    for arm in ARMS:
        center = trajectory[arm]["mean"]
        error = trajectory[arm]["se"]
        axis.plot(
            rounds,
            center,
            label=ARM_LABELS[arm],
            color=COLORS[arm],
            marker=MARKERS[arm],
            linestyle=LINESTYLES[arm],
            linewidth=1.7,
            markersize=4,
        )
        axis.fill_between(
            rounds,
            [value - se for value, se in zip(center, error, strict=True)],
            [value + se for value, se in zip(center, error, strict=True)],
            color=COLORS[arm],
            alpha=0.14,
            linewidth=0,
        )
    axis.axhline(0, color="#999999", linewidth=0.7)
    axis.set_xticks(rounds)
    axis.set_xlabel("Consecutive unsuccessful guess")
    axis.set_ylabel("Change from R1 in frustration-direction\ncosine projection")
    axis.set_title("Five-round frustration-direction trajectory")
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.text(
        1.0,
        0.98,
        f"Complete paired seed blocks: n={len(complete)}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    _save(fig, output / "formal-v2-trajectory")


def _p_text(value: float | None) -> str:
    if value is None:
        return "p_adj unavailable"
    return "p_adj<.001" if value < 0.001 else f"p_adj={value:.3f}"


def _plot_contrasts(output: Path, inference: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.7), constrained_layout=True)
    primary = list(inference["co_primary"])
    derived = inference["derived_total"]
    left_rows = [*primary, derived]
    left_labels = ["Neutral − feedback", "Supportive − neutral", "Supportive − feedback"]
    for index, (row, label) in enumerate(zip(left_rows, left_labels, strict=True)):
        color = "#777777" if row["contrast"] == DERIVED_CONTRAST else "#332288"
        axes[0].errorbar(
            row["mean"],
            index,
            xerr=row["standard_error"],
            fmt="o" if row["contrast"] != DERIVED_CONTRAST else "D",
            color=color,
            capsize=3,
        )
        if row["contrast"] in PRIMARY_CONTRASTS:
            axes[0].annotate(
                _p_text(row["holm_adjusted_p"]),
                (row["mean"], index),
                xytext=(7, 6),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].set_yticks(range(3), left_labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#999999", linewidth=0.7)
    axes[0].set_xlabel("Difference in R1–R5 OLS slope")
    axes[0].set_title("A  Planned arm contrasts")
    axes[0].margins(x=0.25)

    moderation = list(inference["moderation"])
    labels = [
        f"{row['contrast'].replace('_minus_', ' − ').replace('_', ' ')} / {row['effect']}"
        for row in moderation
    ]
    for index, row in enumerate(moderation):
        axes[1].errorbar(
            row["mean"],
            index,
            xerr=row["standard_error"],
            fmt="o",
            color="#117733" if index < 3 else "#CC6677",
            capsize=3,
        )
        axes[1].annotate(
            _p_text(row["holm_adjusted_p"]),
            (row["mean"], index),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#999999", linewidth=0.7)
    axes[1].set_xlabel("Persona moderation of slope contrast")
    axes[1].set_title("B  Planned persona moderation")
    axes[1].margins(x=0.25)
    _save(fig, output / "formal-v2-slope-contrasts")


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}"
    return escape(str(value))


def _table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    headings = "".join(f"<th>{escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(field))}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>"


def _report(output: Path, summary: Mapping[str, Any], complete: Sequence[int]) -> None:
    images = []
    for name in ("formal-v2-trajectory.svg", "formal-v2-slope-contrasts.svg"):
        encoded = base64.b64encode((output / name).read_bytes()).decode("ascii")
        images.append(f'<img alt="{escape(name)}" src="data:image/svg+xml;base64,{encoded}">')
    inference = summary["inference"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>formal-v2 analysis report</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#222}}h1,h2{{font-weight:600}}img{{width:100%;height:auto;margin:12px 0 28px}}table{{border-collapse:collapse;width:100%;margin-bottom:28px}}th,td{{padding:7px 9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{font-family:ui-monospace,monospace}}.note{{color:#555}}
</style></head><body><h1>formal-v2 trajectory analysis</h1>
<p class="note">Complete all-arm seed blocks in trajectory figure: {len(complete)}. Projection values are prompt/context-sensitive and are not subjective experience.</p>
{''.join(images)}
<h2>Co-primary slope contrasts</h2>{_table(inference['co_primary'], ('contrast','count','mean','standard_error','exact_sign_flip_p','holm_adjusted_p'))}
<h2>Persona moderation</h2>{_table(inference['moderation'], ('contrast','effect','count','mean','standard_error','exact_sign_flip_p','holm_adjusted_p'))}
<h2>R5−R1 robustness (descriptive)</h2>{_table(inference['r5_minus_r1_robustness'], ('contrast','effect','count','mean','standard_error'))}
<h2>Manipulation check</h2><p>Passed: <strong>{summary['manipulation']['passed']}</strong>; eligible seeds: {summary['manipulation']['eligible_seed_count']}; positive R5−R1 seeds: {summary['manipulation']['positive_r5_minus_r1_count']}.</p>
</body></html>"""
    (output / "formal-v2-report.html").write_text(html, encoding="utf-8")


def write_figure_bundle(source: Path, destination: Path) -> dict[str, Any]:
    """Create SVG/PDF/PNG figures and a self-contained HTML report."""

    if destination.exists():
        raise FileExistsError(f"figure destination already exists: {destination}")
    summary, rows = _load_bundle(source)
    complete, trajectory = _trajectory(summary, rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="formal-v2-figures-", dir=destination.parent) as raw:
        staging = Path(raw)
        _style()
        _plot_trajectory(staging, complete, trajectory)
        _plot_contrasts(staging, summary["inference"])
        _report(staging, summary, complete)
        hashes = {
            path.name: file_checksum(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "analysis_summary_sha256": file_checksum(source / "summary.json"),
            "complete_all_arm_seeds": complete,
            "artifacts_sha256": hashes,
        }
        (staging / "figure_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    return manifest
