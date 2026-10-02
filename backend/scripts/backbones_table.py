"""
Writes Table 7 (tab:backbones-obs-unobs) into paper/sec/experiment.tex from
data/consol/backbones_3run.json (run backbones_3run.py first). Models with all
three frame runs show mean +/- sd (grey); models still running show their
first run and are shown without a spread until the other runs land. Rows are ordered by the
physics-error question on observable videos.

python backend/scripts/backbones_table.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "data/consol/backbones_3run.json").read_text())
NAME = {"qwen3-vl-32b-instruct": "Qwen3-VL-32B", "qwen3-vl-8b": "Qwen3-VL-8B",
        "qwen2.5-vl-32b": "Qwen2.5-VL-32B", "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
        "internvl3-14b": "InternVL3-14B", "internvl3-8b": "InternVL3-8B",
        "gemma3-12b": "Gemma-3-12B", "llama4-scout-17b": "Llama-4-Scout-17B",
        "llava-ov-7b": "LLaVA-OV-7B", "mistral-small-24b": "Mistral-Small-3.1-24B"}
models = sorted([m for m in R if m in NAME], key=lambda m: -R[m]["mean"][2])
best = np.max([R[m]["mean"] for m in models], axis=0)
sig = R["Signals only"]["mean"]
sig_sd = R["Signals only"].get("sd") or [None] * 3
best = np.maximum(best, [-1, -1, -1, -1, sig[0], sig[1], sig[2]])
pending = [m for m in models if R[m]["n_runs"] < 3]


def cell(v, sd, j):
    c = f"{v:.2f}" + (r"{\scriptsize\textcolor{gray}{$\pm$" + f"{sd:.2f}" + "}}" if sd is not None else "")
    return r"\textbf{" + c + "}" if round(v, 2) >= round(best[j], 2) else c


L = [r"\begin{table}[t]", r"    \centering", r"    \footnotesize", r"    \setlength{\tabcolsep}{3pt}",
     r"    \begin{tabular}{@{}lccccccc@{}}", r"        \toprule",
     r"        & \multicolumn{2}{c}{Standard} & \multicolumn{2}{c}{Physics-error} & \multicolumn{2}{c}{+ signals} & Hidden \\",
     r"        \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-8}",
     r"        VLM & Obs. & Unobs. & Obs. & Unobs. & Obs. & Unobs. & property \\", r"        \midrule"]
for m in models:
    v = R[m]
    full = v["n_runs"] == 3
    vals, sd = (v["mean"], v["sd"]) if full else (v["runs"][0], [None] * 7)
    name = NAME[m]
    L.append(f"        {name} & " + " & ".join(cell(x, s, j) for j, (x, s) in enumerate(zip(vals, sd))) + r" \\")
L += [r"        \midrule",
      r"        Signals only & --- & --- & --- & --- & " + " & ".join(cell(x, sd, j) for j, x, sd in zip((4, 5, 6), sig, sig_sd)) + r" \\",
      r"        \bottomrule", r"    \end{tabular}",
      r"    \caption{Plausibility AUC per VLM on observable ($n{=}320$) and unobservable ($n{=}119$) videos, with a standard question, the PhysicsLENS physics-error question, and that question plus Stage-1/2 signals. \emph{Hidden property}: whether the stated property was followed. Mean $\pm$ sd over three runs that show the VLM different frames of each video (for signals only, three cross-validation splits)"
      + r". Bold: best per column.}",
      r"    \label{tab:backbones-obs-unobs}", r"\end{table}"]
p = ROOT / "paper/sec/experiment.tex"
s = p.read_text()
i = s.index(r"\label{tab:backbones-obs-unobs}")
a = s.rindex(r"\begin{table}[t]", 0, i)
b = s.index(r"\end{table}", i) + len(r"\end{table}")
p.write_text(s[:a] + "\n".join(L) + s[b:])
print(f"Table 7 written: {len(models) - len(pending)} models with 3 runs, pending: {pending}")
