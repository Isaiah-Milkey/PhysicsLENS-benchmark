"""
Stage 4 · Diagnostic Report Generator

First real version:
- Aggregates previous per-test frontend reports.
- Emits a unified JSON report.
- Emits a simple visual global report using Plotly.

Future version:
- Add stronger per-test schemas.
- Add failure-type aggregation.
- Add visual evidence selection.
- Add LLM summary.
- Add PDF/HTML rendering.
"""

import asyncio
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import AsyncGenerator, Any


import plotly.graph_objects as go

from tools.evidence import EVIDENCE, video_id


def _to_float(x, default=None):
    try:
        if isinstance(x, str):
            x = x.replace("%", "").strip()
        return float(x)
    except Exception:
        return default


def _extract_numeric_metrics(test: dict) -> list[dict]:
    out = []
    for m in test.get("metrics", []) or []:
        value = m.get("value")
        num = _to_float(value)
        if num is not None:
            out.append({
                "label": m.get("label", ""),
                "value": num,
                "raw_value": value,
                "sub": m.get("sub", ""),
            })
    return out


def _extract_max_severity(test: dict) -> float | None:
    vals = []
    for s in test.get("severities", []) or []:
        v = _to_float(s.get("value"))
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def _status_to_score(status: str) -> float:
    """
    Very simple placeholder scoring:
    done/success/pass -> good
    error/fail -> bad
    running/unknown -> uncertain
    """
    s = (status or "").lower()
    if s in {"done", "success", "pass"}:
        return 1.0
    if s in {"error", "fail", "failed"}:
        return 0.0
    return 0.5


# ── Semantic findings: VLM-explained violations from the evidence bus ────────
# Stage 3 is now one specialist — a single forced-choice VLM question over
# all seven physics families (+ permanence, + none; see
# pipelines/stage3/specialist_mcq.py) — that writes its structured finding to
# the evidence bus, keyed by video content hash. This is a much richer, more
# reliable "what went wrong" source than parsing the frontend's log lines —
# it's the exact data the specialist's own VLM call produced, not a summary
# of it. (This used to be seven separately-keyed specialists, each with its
# own bus-entry shape; see git history for that version of this function.)
SPECIALIST_DISPLAY = {
    "s3_specialist": "VLM Verification",
}


def _collect_semantic_findings(video_path: str) -> tuple[list[dict], list[dict], list[str]]:
    """Reads the Stage 3 specialist's evidence-bus entry for this exact video
    and returns (semantic_timeline, triage_hypotheses, specialists_seen).

    semantic_timeline: normalized "what went wrong and why" entries with the
    specialist's own explanation. The MCQ judges the whole clip in one call
    (no per-object localization), so every entry has t=None — there is
    exactly one entry per video, present only when the top answer isn't
    "none".
    triage_hypotheses: the Hypothesis Generator's pre-run suspicions (kept
    separate — these are *where the system suspected trouble*, not a
    confirmed finding, and shouldn't be conflated with the timeline above).
    """
    vid = video_id(video_path)
    timeline: list[dict] = []
    specialists_seen: list[str] = []

    for key, name in SPECIALIST_DISPLAY.items():
        ev = EVIDENCE.get(vid, key)
        if not ev:
            continue
        specialists_seen.append(name)

        if ev.get("top_family") and ev["top_family"] != "none":
            timeline.append({
                "source": name, "type": ev["top_family"], "label": None,
                "t": None, "t_end": None,
                "confidence": ev.get("p_violation"),
                "explanation": ev.get("explanation", ""),
                "flagged": True,
            })

    triage: list[dict] = []
    ev_hyp = EVIDENCE.get(vid, "s2_hypothesis_generator")
    if ev_hyp:
        for h in ev_hyp.get("hypotheses", []) or []:
            triage.append({
                "specialist": h.get("specialist"), "confidence": h.get("confidence"),
                "reason": h.get("reason"), "t_window": h.get("t_window"),
            })

    timeline.sort(key=lambda f: (f["t"] is None, f["t"] if f["t"] is not None else 0.0))
    return timeline, triage, specialists_seen


def _build_timeline_plot(semantic_timeline: list[dict]) -> str | None:
    """Scatter of every time-localized finding: x = when, y = which
    specialist, sized/colored by confidence — the visual answer to
    'what went wrong, and when'."""
    dated = [f for f in semantic_timeline if f["t"] is not None]
    if not dated:
        return None

    sources = sorted({f["source"] for f in dated})
    fig = go.Figure()
    palette = ["#c05621", "#1a54c4", "#1a7a3c", "#be185d", "#7c3aed", "#0891b2"]
    for i, src in enumerate(sources):
        pts = [f for f in dated if f["source"] == src]
        fig.add_trace(go.Scatter(
            x=[f["t"] for f in pts], y=[src] * len(pts),
            mode="markers",
            marker=dict(size=[10 + 14 * (f["confidence"] or 0.5) for f in pts],
                       color=palette[i % len(palette)], opacity=0.8,
                       line=dict(color="white", width=1)),
            text=[f"{f['type']}"
                 + (f" ({f['label']})" if f.get("label") else "")
                 + f"<br>{f['explanation']}" for f in pts],
            hovertemplate="t=%{x:.2f}s<br>%{text}<extra></extra>",
            name=src,
        ))
    fig.update_xaxes(title_text="Time (s)", showgrid=True, gridcolor="#ebebeb")
    fig.update_layout(
        title="Timeline of Findings — What Went Wrong, and When",
        height=140 + 50 * max(1, len(sources)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=140, r=40, t=60, b=50),
        showlegend=False,
    )
    return fig.to_json()


def _build_unified_report(previous_results: list[dict], video_name: str,
                          semantic_timeline: list[dict] | None = None,
                          triage_hypotheses: list[dict] | None = None,
                          specialists_with_evidence: list[str] | None = None) -> dict:
    included = [
        r for r in previous_results
        if not r.get("excluded", False) and r.get("pipelineId") != "s4_report"
    ]

    by_stage = defaultdict(list)
    for r in included:
        by_stage[str(r.get("stageName", "Unknown Stage"))].append(r)

    status_counts = Counter((r.get("status") or "unknown") for r in included)

    severity_values = []
    local_reports = []

    for r in included:
        max_sev = _extract_max_severity(r)
        if max_sev is not None:
            severity_values.append(max_sev)

        numeric_metrics = _extract_numeric_metrics(r)

        warnings = [
            l.get("text", "")
            for l in r.get("logs", []) or []
            if l.get("level") == "warn"
        ]

        errors = [
            l.get("text", "")
            for l in r.get("logs", []) or []
            if l.get("level") == "error"
        ]

        local_reports.append({
            "test_id": r.get("id"),
            "stage": {
                "id": r.get("stageId"),
                "name": r.get("stageName"),
                "num": r.get("stageNum"),
            },
            "pipeline": {
                "id": r.get("pipelineId"),
                "name": r.get("pipelineName"),
            },
            "status": r.get("status"),
            "timestamp": r.get("timestamp"),
            "max_severity": max_sev,
            "numeric_metrics": numeric_metrics,
            "num_metrics": len(r.get("metrics", []) or []),
            "num_severity_items": len(r.get("severities", []) or []),
            "num_warnings": len(warnings),
            "num_errors": len(errors),
            "warnings": warnings[:5],
            "errors": errors[:5],
            "user_comment": r.get("comment", ""),
        })

    if severity_values:
        avg_severity = sum(severity_values) / len(severity_values)
        max_severity = max(severity_values)
    else:
        avg_severity = None
        max_severity = None

    status_score = (
        sum(_status_to_score(r.get("status")) for r in included) / len(included)
        if included else None
    )

    # Placeholder global score.
    # Later, replace this with a real physics scoring model.
    if status_score is None:
        physics_consistency_score = None
    elif avg_severity is None:
        physics_consistency_score = round(status_score * 100, 2)
    else:
        physics_consistency_score = round(
            max(0.0, min(100.0, status_score * 100.0 - avg_severity * 0.35)),
            2,
        )

    if max_severity is None:
        global_severity_label = "unknown"
    elif max_severity >= 75:
        global_severity_label = "critical"
    elif max_severity >= 50:
        global_severity_label = "high"
    elif max_severity >= 25:
        global_severity_label = "medium"
    else:
        global_severity_label = "low"

    semantic_timeline = semantic_timeline or []
    triage_hypotheses = triage_hypotheses or []
    specialists_with_evidence = specialists_with_evidence or []

    recommended_followup = []
    if not included:
        recommended_followup.append("Run at least one Stage 1 or Stage 2 diagnostic test before generating the final report.")
    if any(r["num_errors"] > 0 for r in local_reports):
        recommended_followup.append("Inspect tests with backend errors before trusting the final diagnosis.")
    if max_severity is not None and max_severity >= 50:
        recommended_followup.append("Run localized and specialist diagnostics around the high-severity event windows.")
    if len(included) < 2:
        recommended_followup.append("Run multiple complementary tests to improve confidence.")
    if not semantic_timeline:
        untried = [n for n in SPECIALIST_DISPLAY.values() if n not in specialists_with_evidence]
        if untried:
            recommended_followup.append(
                "No Stage 3 specialist evidence found for this video yet — run "
                + ", ".join(untried) + " for a semantic (what/when/why) diagnosis, "
                "not just numeric severities."
            )

    # Richer basis for confidence when specialist VLM evidence backs the report.
    if semantic_timeline:
        confidence_level = "medium" if len(specialists_with_evidence) < 2 else "high"
        confidence_reason = (
            f"Backed by {len(semantic_timeline)} VLM-explained finding(s) from "
            f"{len(specialists_with_evidence)} Stage 3 specialist(s) "
            f"({', '.join(specialists_with_evidence)}), not just numeric severities."
        )
    else:
        confidence_level = "low" if len(included) < 2 else "medium"
        confidence_reason = "This is a first-pass aggregation based only on available frontend test outputs."

    report = {
        "schema_version": "0.1",
        "report_type": "physicslens_global_diagnosis",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": video_name,

        "global_summary": {
            "num_tests_total": len(previous_results),
            "num_tests_included": len(included),
            "status_counts": dict(status_counts),
            "physics_consistency_score": physics_consistency_score,
            "global_severity": {
                "label": global_severity_label,
                "max_value": max_severity,
                "average_value": round(avg_severity, 2) if avg_severity is not None else None,
            },
            "confidence": {
                "level": confidence_level,
                "reason": confidence_reason,
            },
        },

        "stage_summary": {
            stage: {
                "num_tests": len(items),
                "pipelines": [x.get("pipelineName") for x in items],
                "statuses": dict(Counter(x.get("status") or "unknown" for x in items)),
            }
            for stage, items in by_stage.items()
        },

        "local_reports": local_reports,

        # The semantic "what and when could have gone wrong" evidence: each
        # entry is a Stage 3 specialist's own VLM-explained finding (source,
        # type, timestamp, confidence, plain-language explanation), pulled
        # directly from the evidence bus rather than reconstructed from logs.
        "semantic_timeline": semantic_timeline,
        "specialists_with_evidence": specialists_with_evidence,
        # Where the automated Hypothesis Generator suspected trouble BEFORE
        # specialists ran — background context, not a confirmed finding.
        "triage_hypotheses": triage_hypotheses,

        "recommended_followup": recommended_followup,

        "llm_ready_context": {
            "instruction": (
                "Summarize the global physics diagnosis using semantic_timeline as "
                "the primary source for WHAT went wrong and WHEN, cross-checked "
                "against global_summary/local_reports for overall severity and "
                "confidence, and recommended follow-up tests."
            ),
            "inputs": {
                "global_summary": "See global_summary.",
                "stage_summary": "See stage_summary.",
                "local_reports": "See local_reports.",
                "semantic_timeline": (
                    "Chronological VLM-explained findings — the ground truth for "
                    "'what and when'. See semantic_timeline."
                ),
                "triage_hypotheses": (
                    "Pre-run suspicions from the Hypothesis Generator — context "
                    "only, not confirmed findings. See triage_hypotheses."
                ),
            },
        },
    }

    return report


def _build_visual_summary(report: dict) -> str:
    local_reports = report.get("local_reports", [])

    names = []
    severities = []

    for r in local_reports:
        pipeline_name = r.get("pipeline", {}).get("name", "Unknown test")
        stage_name = r.get("stage", {}).get("name", "Unknown stage")
        max_sev = r.get("max_severity")

        if max_sev is None:
            max_sev = 0

        names.append(f"{pipeline_name}<br><sup>{stage_name}</sup>")
        severities.append(max_sev)

    if not names:
        names = ["No previous tests"]
        severities = [0]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=severities,
        text=[f"{v:.1f}%" for v in severities],
        textposition="auto",
        name="Max severity",
    ))

    fig.update_layout(
        title="Global Diagnostic Report: Severity by Test",
        xaxis_title="Diagnostic Test",
        yaxis_title="Max Severity (%)",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=20, t=60, b=120),
        height=420,
    )

    return fig.to_json()

def _build_llm_prompt(report: dict) -> str:
    """
    Convert the unified PhysicsLENS JSON report into a prompt for the LLM.

    The "Timeline of Findings" narrative only exists when Stage 3 specialists
    have produced explained findings. When `semantic_timeline` is empty we drop
    that section from the requested format entirely — otherwise the model writes
    an awkward "no findings yet" placeholder — and instead tell it to ground the
    diagnosis in the numeric severities/metrics.
    """
    report_text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    n_findings = len(report.get("semantic_timeline") or [])
    has_timeline = n_findings > 0

    if has_timeline:
        timeline_guidance = (
            "`semantic_timeline` is the ground truth for this: each entry is one "
            "Stage 3 specialist's confirmed finding, already timestamped and "
            "explained in plain language by that specialist's own vision-model "
            "check (e.g. \"at t=0.85s, the ball's momentum vanishes with no "
            "visible cause\"). Walk through it in chronological order and narrate "
            "it as a coherent story of the video's physical failures — do not "
            "just list the entries verbatim, connect them (e.g. a collision "
            "finding at t=2.1s and a deformation finding at t=2.15s are probably "
            "the same event described by two specialists).\n\n"
        )
        timeline_rule = (
            "- Every specific claim about what went wrong and when must trace to "
            "an entry in semantic_timeline (cite its timestamp) or a metric in "
            "local_reports — never state a failure that isn't backed by one of "
            "these.\n"
        )
        timeline_section = (
            "\n## Timeline of Findings\n"
            "(walk through semantic_timeline chronologically — what happened, "
            f"when, per which specialist, and why it's a violation; {n_findings} "
            "finding(s) available.)\n"
        )
    else:
        timeline_guidance = ""
        timeline_rule = (
            "- No Stage 3 specialist has produced an explained finding for this "
            "video, so there is NO timeline to narrate. Do not add a \"Timeline "
            "of Findings\" section and do not speculate one. Base the diagnosis "
            "on the numeric severities/metrics in global_summary and "
            "local_reports, and note plainly that these are numeric-only signals "
            "without a semantic what/when explanation.\n"
        )
        timeline_section = ""

    return f"""
You are a physics diagnostic assistant for AI-generated videos.

You will receive a structured JSON report from PhysicsLENS. Your job is to
write a concise final diagnosis for a human evaluator that explains, in plain
language, WHAT physically went wrong in this video and WHEN it happened —
grounded in the specialists' own VLM-generated explanations, not just scores.

{timeline_guidance}`triage_hypotheses` (if present) is where the automated system suspected
trouble BEFORE the specialists ran — cite it only as background on why those
specialists were run, never as a confirmed finding.

Important rules:
- Do not invent evidence that is not present in the JSON.
{timeline_rule}- If a score is described as prototype, first-pass, or placeholder, say that clearly.
- Distinguish between "no issue detected" and "no severity data reported".
- Use careful language such as "the available diagnostics suggest" rather than overconfident claims.

Please return the answer in this format:

# PhysicsLENS Final Diagnosis

## Overall Assessment
...
{timeline_section}
## Main Evidence
...

## Severity and Confidence
...

## Recommended Follow-up
...

Here is the PhysicsLENS JSON report:

{report_text}
""".strip()


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}

    output_format = cfg.get("output_format", "json_visual")
    previous_results = cfg.get("previous_results", [])
    video_name = cfg.get("video_name", "uploaded video")
    use_llm_summary = str(cfg.get("use_llm_summary", "false")).lower() == "true"
    api_key = str(cfg.get("api_key", "")).strip()
    summary_model = str(cfg.get("summary_model") or "gpt-4o-mini").strip()

    yield {
        "type": "log",
        "level": "info",
        "text": f"Diagnostic Report Generator received {len(previous_results)} previous test result(s).",
    }
    await asyncio.sleep(0)

    semantic_timeline, triage_hypotheses, specialists_with_evidence = \
        _collect_semantic_findings(video_path)
    if semantic_timeline:
        yield {
            "type": "log", "level": "info",
            "text": (f"Evidence bus: {len(semantic_timeline)} VLM-explained finding(s) "
                     f"from {len(specialists_with_evidence)} specialist(s) "
                     f"({', '.join(specialists_with_evidence)})."),
        }
    else:
        yield {
            "type": "log", "level": "warn",
            "text": ("No Stage 3 specialist evidence found on the bus for this video — "
                     "the report will be numeric-only (severities/metrics), not a "
                     "semantic what/when narrative. Run some Stage 3 specialists first."),
        }
    await asyncio.sleep(0)

    report = _build_unified_report(previous_results, video_name,
                                   semantic_timeline, triage_hypotheses,
                                   specialists_with_evidence)

    score = report["global_summary"]["physics_consistency_score"]
    sev = report["global_summary"]["global_severity"]

    yield {
        "type": "metric",
        "label": "Included tests",
        "value": report["global_summary"]["num_tests_included"],
        "sub": "Previous non-excluded test results aggregated into this report.",
    }

    yield {
        "type": "metric",
        "label": "Physics consistency score",
        "value": "N/A" if score is None else score,
        "sub": "Placeholder score for first-pass testing.",
    }

    yield {
        "type": "metric",
        "label": "Semantic findings",
        "value": len(semantic_timeline),
        "sub": (f"VLM-explained, from {', '.join(specialists_with_evidence)}"
               if specialists_with_evidence else "no Stage 3 specialist evidence yet"),
    }

    if sev["max_value"] is not None:
        color = "#dc2626" if sev["max_value"] >= 75 else "#f97316" if sev["max_value"] >= 50 else "#facc15" if sev["max_value"] >= 25 else "#22c55e"
        yield {
            "type": "severity",
            "label": f"Global severity: {sev['label']}",
            "value": round(sev["max_value"], 2),
            "color": color,
        }

    if output_format in {"json_visual", "json"}:
        yield {
            "type": "result",
            "title": "Unified Diagnostic JSON",
            "report": report,
        }

    timeline_plot = _build_timeline_plot(semantic_timeline)
    if output_format == "json_visual" and timeline_plot:
        yield {
            "type": "plotly",
            "caption": "What went wrong, and when — one marker per VLM-confirmed "
                      "finding across all Stage 3 specialists that ran.",
            "data": timeline_plot,
        }

    if output_format == "json_visual":
        yield {
            "type": "plotly",
            "caption": "Global visual summary",
            "data": _build_visual_summary(report),
        }

    if use_llm_summary:
        yield {
            "type": "log",
            "level": "info",
            "text": f"Requesting LLM summary from OpenAI ({summary_model}).",
        }

        try:
            from tools.llm_api import query_text, response_text
            data = await query_text(_build_llm_prompt(report),
                                    model=summary_model,
                                    token=api_key or None)
            llm_summary = response_text(data) or json.dumps(data, indent=2)

            yield {
                "type": "log",
                "level": "info",
                "text": f"OpenAI responded ({summary_model}). Summary length: {len(llm_summary)} characters.",
            }

            report["llm_summary"] = {
                "provider": "OpenAI",
                "model": summary_model,
                "summary": llm_summary,
            }

            yield {
                "type": "llm_summary",
                "title": "LLM Diagnostic Summary",
                "markdown": llm_summary,
            }

        except Exception as e:
            yield {
                "type": "log",
                "level": "error",
                "text": f"LLM summary failed: {e}",
            }

    yield {"type": "done"}