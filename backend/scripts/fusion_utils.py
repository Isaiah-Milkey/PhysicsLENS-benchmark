"""Fusion of a VLM score with Stage-1/2 signals, weight fitted on training folds
(nested). Shared by main_table_full.py and leader_search2.py."""
import numpy as np
from system_eval import tool_oof, rank01, spear
from specialist_accuracy import auc

GRID = np.linspace(0, 1, 11)


def make_fused(X, G, F, N):
    def fused(v, target, ybin, mask=None):
        """Out-of-fold fusion of VLM score v with Stage-1/2 signals, weight fitted
        on training folds. target: what the signals are selected against;
        ybin: binary label the weight is tuned for. Returns a score for every clip
        in mask (NaN elsewhere)."""
        mask = np.ones(N, bool) if mask is None else mask
        out = np.full(N, np.nan)
        rv = rank01(v)
        for i in range(5):
            tr, te = mask & (F != i), mask & (F == i)
            if te.sum() == 0:
                continue
            # signals for the TEST fold: selected on all training folds
            t_sel = tool_oof_fit_apply(X[tr], target[tr], X[te])
            # signals for the TRAINING folds, out-of-fold within them (nested)
            t_tr = tool_oof(X[tr], target[tr], G[tr])
            rt_tr, rv_tr = rank01(t_tr), rank01(v[tr])
            yb = ybin[tr]
            best = max(GRID, key=lambda w: auc(yb, w * rv_tr + (1 - w) * rt_tr))
            # rank the test fold's signal score against the training folds' scores
            ref = np.sort(tool_oof_fit_apply(X[tr], target[tr], X[tr]))
            rt_te = np.searchsorted(ref, t_sel) / max(len(ref) - 1, 1)
            rv_te = np.searchsorted(np.sort(v[tr]), v[te]) / max(tr.sum() - 1, 1)
            out[te] = best * rv_te + (1 - best) * rt_te
        return out


    def tool_oof_fit_apply(Xtr, ytr, Xte, k=5):
        """Select + orient the top-k signals on (Xtr, ytr); score Xte with them.
        Ranks are taken relative to the training rows."""
        from system_eval import spear
        R = np.column_stack([rank01(Xtr[:, j]) for j in range(Xtr.shape[1])])
        ok = np.isfinite(ytr)
        r = np.nan_to_num(np.array([spear(R[ok, j], ytr[ok]) for j in range(R.shape[1])]))
        top = np.argsort(-np.abs(r))[:k]
        cols = []
        for j in top:
            ref = np.sort(Xtr[:, j])
            cols.append(np.sign(r[j]) * np.searchsorted(ref, Xte[:, j]) / max(len(ref) - 1, 1))
        return np.mean(cols, axis=0)



    return fused
