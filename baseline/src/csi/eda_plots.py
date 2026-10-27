"""All stage-1 EDA figures (project spec, section 6.2) and the threshold baseline (6.3).

Usage: python -m csi.eda_plots [--dataset wallhack|3do|all]
Figures go to figures/, baseline tables to results/.
Needs data/processed/<dataset>/{recordings,windows}.parquet (python -m csi.format; python -m csi.eda).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from csi import features as F
from csi.eda import load_windows
from csi.format import ROOT, load_index, load_recording
from csi.labels import DATASET_TO_GLOBAL
from csi.plotstyle import DOMAIN_COLORS, INK2, LABEL_COLORS, MUTED, SEQ, plt
from csi.windows import WINDOW

FIG = ROOT / "figures"
RES = ROOT / "results"
FS = F.FS
SHOW_SC = [3, 16, 30, 44]  # columns of the 52 used for line plots (subcarriers -23, -10, 5, 19)

WALLHACK_DOMAINS = ["LOS_BQ", "LOS_PIFA", "NLOS_BQ", "NLOS_PIFA"]
DAYS = ["d1", "d2", "d3"]


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name)


def label_legend(ax, labels, **kw):
    ax.legend(handles=[Patch(color=LABEL_COLORS[lab], label=lab) for lab in labels], **kw)


# ---------------------------------------------------------------- recording picks

def examples(dataset: str, domain: str) -> dict[str, tuple[str, slice]]:
    """{label: (recording_id, packet slice)} — one 40 s segment per class for a domain/day."""
    seg = int(40 * FS)
    if dataset == "wallhack":
        base = f"wallhack_{domain}_"
        picks = {"empty": base + "b1", "walk": base + "w1", "walk_wave": base + "ww1"}
        out = {}
        for lab, rid in picks.items():
            n = len(load_recording(dataset, rid)["t"])
            mid = n // 2
            out[lab] = (rid, slice(mid - seg // 2, mid + seg // 2))
        return out
    out = {}
    for lab, rid, cls in [("empty", f"3do_{domain}_b1", 0), ("walk", f"3do_{domain}_w1", 1),
                          ("sit", f"3do_{domain}_s1", 2), ("lie", f"3do_{domain}_l1", 3)]:
        y = load_recording(dataset, rid)["y"]
        idx = np.flatnonzero(y == cls)
        mid = idx[len(idx) // 2]
        out[lab] = (rid, slice(mid - seg // 2, mid + seg // 2))
    return out


def domains(dataset):
    return WALLHACK_DOMAINS if dataset == "wallhack" else DAYS


# ---------------------------------------------------------------- 6.1 data checks

def plot_dt_hist(dataset):
    idx = load_index(dataset)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(0, 40, 161)
    for dom in domains(dataset):
        dts = []
        for r in idx.itertuples():
            if (f"{r.condition}_{r.antenna}" if dataset == "wallhack" else r.day) != dom:
                continue
            dts.append(np.diff(load_recording(dataset, r.recording_id)["t"]) * 1e3)
        dt = np.concatenate(dts)
        ax.hist(np.clip(dt, 0, 40), bins=bins, histtype="step", lw=1.4, color=DOMAIN_COLORS[dom],
                label=f"{dom}: median {np.median(dt):.2f} ms, >30 ms {np.mean(dt > 30) * 100:.2f}%")
    ax.set_yscale("log")
    ax.set_xlabel("interval between packets, ms (clipped at 40)")
    ax.set_ylabel("packets")
    ax.set_title(f"{dataset}: inter-packet intervals (nominal 10 ms = 100 Hz)")
    ax.legend(fontsize=8)
    save(fig, f"{dataset}_dt_hist.png")


def plot_overview(dataset):
    """Mean amplitude and RSSI over every full recording: AGC steps, drifts, outliers."""
    idx = load_index(dataset)
    n = len(idx)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 1.5), sharey=False)
    for ax, r in zip(axes.flat, idx.itertuples()):
        d = load_recording(dataset, r.recording_id)
        a = F.amplitude(d["csi"]).mean(axis=1)
        ax.plot(d["t"], F.rolling_mean(a, 50), lw=0.7, color=INK2)
        ax.set_title(r.recording_id.replace(f"{dataset}_", ""), fontsize=7, fontweight="normal")
        ax.tick_params(labelsize=6)
        ax.grid(False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{dataset}: mean amplitude over each full recording (0.5 s smoothing), x = s")
    fig.tight_layout()
    save(fig, f"{dataset}_amp_overview_all_recordings.png")


def amp_level_steps(dataset) -> pd.DataFrame:
    """Detect AGC-like steps: jumps of the 2 s median level larger than 15% between adjacent seconds."""
    out = []
    for r in load_index(dataset).itertuples():
        d = load_recording(dataset, r.recording_id)
        a = F.amplitude(d["csi"]).mean(axis=1)
        n = len(a) // 100
        sec = np.median(a[: n * 100].reshape(n, 100), axis=1)
        lvl = pd.Series(sec).rolling(2, center=True, min_periods=1).median().to_numpy()
        rel = np.abs(np.diff(lvl)) / np.maximum(lvl[:-1], 1e-6)
        zeros = (F.amplitude(d["csi"]) == 0).mean(axis=0)
        out.append({"recording_id": r.recording_id, "n_level_steps_15pct": int((rel > 0.15).sum()),
                    "max_rel_step": float(rel.max()) if len(rel) else 0.0,
                    "amp_mean": float(a.mean()), "amp_cv_over_time": float(sec.std() / sec.mean()),
                    "max_zero_frac_subcarrier": float(zeros.max())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 6.2.1 / 6.2.2

def plot_heatmaps_and_lines(dataset):
    for dom in domains(dataset):
        ex = examples(dataset, dom)
        segs = {lab: (rid, F.amplitude(load_recording(dataset, rid)["csi"][sl])) for lab, (rid, sl) in ex.items()}
        allv = np.concatenate([a.ravel() for _, a in segs.values()])
        vmin, vmax = np.percentile(allv, [1, 99])
        fig, axes = plt.subplots(1, len(segs), figsize=(4 * len(segs), 3.2), sharey=True)
        for ax, (lab, (rid, a)) in zip(axes, segs.items()):
            im = ax.imshow(a.T, aspect="auto", origin="lower", cmap=SEQ, vmin=vmin, vmax=vmax,
                           extent=[0, len(a) / FS, 0, a.shape[1]], interpolation="nearest")
            ax.set_title(f"{lab}  ({rid.split('_')[-1]})")
            ax.set_xlabel("time, s")
            ax.grid(False)
        axes[0].set_ylabel("subcarrier (column 0..51)")
        fig.colorbar(im, ax=axes, label="|CSI|", shrink=0.9)
        fig.suptitle(f"{dataset} {dom}: amplitude, subcarriers x time, one color scale", y=1.02)
        save(fig, f"{dataset}_{dom}_amp_heatmap.png")

        fig, axes = plt.subplots(len(segs), 1, figsize=(9, 1.7 * len(segs)), sharex=True, sharey=True)
        for ax, (lab, (rid, a)) in zip(axes, segs.items()):
            t = np.arange(len(a)) / FS
            for j, sc in enumerate(SHOW_SC):
                ax.plot(t, a[:, sc], lw=0.8, color=plt.cm.Greys(0.45 + 0.15 * j),
                        label=f"sc {int(F.SUBCARRIER_NUMBERS[sc])}")
            ax.set_ylabel(lab, color=LABEL_COLORS[lab], fontweight="bold")
        axes[0].legend(ncol=4, fontsize=7, loc="upper right")
        axes[-1].set_xlabel("time, s")
        fig.suptitle(f"{dataset} {dom}: amplitude of 4 subcarriers over time")
        fig.tight_layout()
        save(fig, f"{dataset}_{dom}_amp_lines.png")


# ---------------------------------------------------------------- 6.2.3 rolling variance over time

def _label_band(ax, t, y, dataset):
    names = DATASET_TO_GLOBAL[dataset]
    change = np.flatnonzero(np.diff(y)) + 1
    bounds = np.r_[0, change, len(y)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax.axvspan(t[a], t[b - 1], color=LABEL_COLORS[names[int(y[a])]], lw=0)
    ax.set_yticks([])
    ax.grid(False)


def plot_rolling_var_time(dataset):
    if dataset == "wallhack":
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        for ax, dom in zip(axes, WALLHACK_DOMAINS):
            t0 = 0.0
            for lab, (rid, sl) in examples(dataset, dom).items():
                d = load_recording(dataset, rid)
                a = F.amplitude(d["csi"])
                ms = F.motion_score(a, 100)[sl]
                t = t0 + np.arange(len(ms)) / FS
                ax.plot(t, ms, lw=0.8, color=LABEL_COLORS[lab])
                ax.axvspan(t[0], t[-1], ymin=0, ymax=0.04, color=LABEL_COLORS[lab], lw=0)
                t0 = t[-1] + 2
            ax.set_yscale("log")
            ax.set_ylabel(f"{dom}\nvar (1 s)")
        label_legend(axes[0], ["empty", "walk", "walk_wave"], ncol=3, loc="upper left")
        axes[-1].set_xlabel("time, s (three 40 s segments: b1 | w1 | ww1, different files)")
        fig.suptitle("wallhack: rolling variance of amplitude (1 s window, mean over subcarriers)")
        fig.tight_layout()
        save(fig, "wallhack_rollvar_time.png")
        return
    for day in DAYS:
        rids = [f"3do_{day}_b1", f"3do_{day}_w1", f"3do_{day}_s1", f"3do_{day}_l1"]
        fig, axes = plt.subplots(len(rids) * 2, 1, figsize=(11, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [5, 0.6] * len(rids)})
        for i, rid in enumerate(rids):
            d = load_recording(dataset, rid)
            ms = F.motion_score(F.amplitude(d["csi"]), 100)
            ax, band = axes[2 * i], axes[2 * i + 1]
            ax.plot(d["t"], ms, lw=0.5, color=INK2)
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 3e2)
            ax.set_ylabel(rid.split("_")[-1])
            _label_band(band, d["t"], d["y"], dataset)
        label_legend(axes[0], ["empty", "walk", "sit", "lie"], ncol=4, loc="upper right")
        axes[-1].set_xlabel("time, s")
        fig.suptitle(f"3do {day}: rolling variance of amplitude (1 s, mean over subcarriers); "
                     "color band = per-packet label")
        fig.tight_layout()
        save(fig, f"3do_{day}_rollvar_time.png")


# ---------------------------------------------------------------- 6.2.4 distributions

def plot_distributions(dataset, w: pd.DataFrame):
    labels = list(DATASET_TO_GLOBAL[dataset].values())
    doms = domains(dataset)
    for feat, title in [("var1s", "variance of amplitude (1 s, mean over subcarriers)"),
                        ("cv1s", "normalized variance (var / mean², 1 s)"),
                        ("var1s_clean", "variance of amplitude after impulse removal (1 s)"),
                        ("impulse_frac", "share of impulsive (drop-out) packets"),
                        ("phase_var1s", "variance of sanitized phase (1 s)")]:
        fig, ax = plt.subplots(figsize=(2.6 * len(doms) + 1, 3.6))
        width = 0.8 / len(labels)
        for i, dom in enumerate(doms):
            for j, lab in enumerate(labels):
                v = w.loc[(w.domain == dom) & (w.label == lab), feat]
                v = v * 100 if feat == "impulse_frac" else np.log10(v)
                if len(v) == 0:
                    continue
                pos = i + (j - (len(labels) - 1) / 2) * width
                bp = ax.boxplot(v, positions=[pos], widths=width * 0.8, patch_artist=True,
                                showfliers=False, medianprops={"color": "white", "lw": 1.5})
                bp["boxes"][0].set(facecolor=LABEL_COLORS[lab], edgecolor=LABEL_COLORS[lab])
                for k in ("whiskers", "caps"):
                    for line in bp[k]:
                        line.set(color=LABEL_COLORS[lab])
        ax.set_xticks(range(len(doms)), doms)
        ax.set_xlim(-0.6, len(doms) - 0.4)
        ax.set_ylabel("% of packets" if feat == "impulse_frac" else f"log10 {feat}")
        ax.set_title(f"{dataset}: {title}, per 4 s window")
        label_legend(ax, labels, ncol=len(labels), loc="upper left", fontsize=8)
        save(fig, f"{dataset}_{feat}_dist.png")


# ---------------------------------------------------------------- 6.2.5 phase

def plot_phase(dataset):
    dom = "NLOS_BQ" if dataset == "wallhack" else "d1"
    ex = examples(dataset, dom)
    labs = [lab for lab in ex if lab in ("empty", "walk")]
    seg = int(10 * FS)
    fig, axes = plt.subplots(2, len(labs), figsize=(5 * len(labs), 5), sharex=True)
    for j, lab in enumerate(labs):
        rid, sl = ex[lab]
        csi = load_recording(dataset, rid)["csi"][sl][:seg]
        t = np.arange(len(csi)) / FS
        raw, san = F.raw_phase(csi), F.sanitized_phase(csi)
        for k, sc in enumerate(SHOW_SC[:3]):
            c = plt.cm.Greys(0.5 + 0.2 * k)
            axes[0, j].plot(t, raw[:, sc], lw=0.6, color=c, label=f"sc {int(F.SUBCARRIER_NUMBERS[sc])}")
            axes[1, j].plot(t, san[:, sc], lw=0.8, color=c)
        axes[0, j].set_title(f"{lab}: raw phase", color=LABEL_COLORS[lab])
        axes[1, j].set_title(f"{lab}: sanitized phase", color=LABEL_COLORS[lab])
        axes[1, j].set_xlabel("time, s")
    axes[0, 0].set_ylabel("rad")
    axes[1, 0].set_ylabel("rad")
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"{dataset} {dom}: phase before / after sanitization "
                 "(unwrap over subcarriers, minus per-packet linear fit)")
    fig.tight_layout()
    save(fig, f"{dataset}_{dom}_phase.png")


def plot_pairing_comparison():
    """The byte-pairing question (docs/DECISIONS.md #1): profiles and phase for both pairings."""
    rid = "3do_d1_b1"
    d = load_recording("3do", rid)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    k = F.SUBCARRIER_NUMBERS
    for pairing, color in [("authors", "#eb6834"), ("espidf", "#2a78d6")]:
        from csi.io.esp32 import csi_from_raw
        csi = csi_from_raw(d["raw"][:3000], pairing)
        a = np.abs(csi)
        axes[0].plot(k, a.mean(axis=0), "o-", ms=3, color=color, label=pairing)
        axes[1].plot(k, a[100], "o-", ms=3, color=color, label=pairing)
        p = np.unwrap(np.angle(csi[100]))
        axes[2].plot(k, p - p.mean(), "o-", ms=3, color=color, label=pairing)
    axes[0].set_title("mean amplitude profile (30 s)")
    axes[1].set_title("amplitude, one packet")
    axes[2].set_title("unwrapped phase, one packet")
    for ax in axes:
        ax.set_xlabel("subcarrier number")
        ax.legend(fontsize=8)
    fig.suptitle(f"Byte pairing: authors' loader vs ESP-IDF layout ({rid})")
    fig.tight_layout()
    save(fig, "pairing_authors_vs_espidf.png")


# ---------------------------------------------------------------- 6.2.6 spectra

def plot_spectra(dataset):
    for dom in domains(dataset)[:1] if dataset == "3do" else domains(dataset):
        ex = examples(dataset, dom)
        fig, axes = plt.subplots(1, len(ex), figsize=(4 * len(ex), 3.2), sharey=True)
        imgs = []
        for ax, (lab, (rid, sl)) in zip(axes, ex.items()):
            a, _ = F.remove_impulses(F.amplitude(load_recording(dataset, rid)["csi"]))
            pc = F.first_pc(a)[sl]
            f, t, z = F.stft_db(pc)
            imgs.append((ax, lab, f, t, z))
        vmax = max(np.percentile(z, 99) for *_, z in imgs)
        for ax, lab, f, t, z in imgs:
            im = ax.pcolormesh(t, f, z, cmap=SEQ, vmin=vmax - 50, vmax=vmax, shading="auto")
            ax.set_title(lab, color=LABEL_COLORS[lab])
            ax.set_xlabel("time, s")
            ax.grid(False)
        axes[0].set_ylabel("frequency, Hz")
        fig.colorbar(im, ax=axes, label="dB", shrink=0.9)
        fig.suptitle(f"{dataset} {dom}: STFT of 1st PCA component of amplitude after impulse removal "
                     "(power changes, not true Doppler)", y=1.02)
        save(fig, f"{dataset}_{dom}_stft.png")

    for variant in ("raw", "clean"):
        plot_psd(dataset, variant)


def plot_psd(dataset, variant):
    """PSD per class, median over 4 s windows, one panel per domain."""
    doms = domains(dataset)
    fig, axes = plt.subplots(1, len(doms), figsize=(3.6 * len(doms), 3.2), sharey=True)
    idx = load_index(dataset)
    names = DATASET_TO_GLOBAL[dataset]
    for ax, dom in zip(axes, doms):
        acc: dict[str, list] = {}
        for r in idx.itertuples():
            if (f"{r.condition}_{r.antenna}" if dataset == "wallhack" else r.day) != dom:
                continue
            d = load_recording(dataset, r.recording_id)
            amp = F.amplitude(d["csi"])
            pcs = {"raw": F.first_pc(amp), "clean": F.first_pc(F.remove_impulses(amp)[0])}
            pc = pcs[variant]
            for s in range(0, len(pc) - WINDOW + 1, WINDOW):
                vals, cnt = np.unique(d["y"][s:s + WINDOW], return_counts=True)
                if cnt.max() < 0.8 * WINDOW:
                    continue
                f, p = F.psd(pc[s:s + WINDOW], nperseg=256)
                acc.setdefault(names[int(vals[cnt.argmax()])], []).append(p)
        for lab in names.values():
            if lab in acc:
                ax.semilogy(f, np.median(acc[lab], axis=0), color=LABEL_COLORS[lab], lw=1.4,
                            label=f"{lab} (n={len(acc[lab])})")
        ax.set_title(dom)
        ax.set_xlabel("Hz")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("PSD of PC1 (median over 4 s windows)")
    fig.suptitle(f"{dataset}: power spectral density by class "
                 f"({'raw amplitude' if variant == 'raw' else 'after impulse removal'})")
    fig.tight_layout()
    save(fig, f"{dataset}_psd_by_class_{variant}.png")


# ---------------------------------------------------------------- 6.2.7 domain comparison

def plot_domain_profiles(dataset):
    idx = load_index(dataset)
    names = DATASET_TO_GLOBAL[dataset]
    doms = domains(dataset)
    k = F.SUBCARRIER_NUMBERS
    fig, axes = plt.subplots(1, len(doms) + 1, figsize=(3.4 * (len(doms) + 1), 3.2))
    profiles = {}
    for ax, dom in zip(axes, doms):
        for r in idx.itertuples():
            if (f"{r.condition}_{r.antenna}" if dataset == "wallhack" else r.day) != dom:
                continue
            d = load_recording(dataset, r.recording_id)
            a = F.amplitude(d["csi"])
            for c in np.unique(d["y"]):
                prof = a[d["y"] == c].mean(axis=0)
                lab = names[int(c)]
                profiles.setdefault((dom, lab), []).append(prof)
                ax.plot(k, prof, lw=0.8, alpha=0.8, color=LABEL_COLORS[lab])
        ax.set_title(dom)
        ax.set_xlabel("subcarrier number")
    axes[0].set_ylabel("mean |CSI| per recording")
    label_legend(axes[0], list(names.values()), fontsize=7)
    # summary panel: mean profile per domain (empty only) -> how much the room changes the signal
    ax = axes[-1]
    for dom in doms:
        if (dom, "empty") in profiles:
            ax.plot(k, np.mean(profiles[(dom, "empty")], axis=0), lw=1.6, color=DOMAIN_COLORS[dom], label=dom)
    ax.set_title("empty room, by domain")
    ax.set_xlabel("subcarrier number")
    ax.legend(fontsize=7)
    fig.suptitle(f"{dataset}: mean amplitude profile over subcarriers (one line = one recording x class)")
    fig.tight_layout()
    save(fig, f"{dataset}_domain_profiles.png")


def domain_vs_class_distance(w: pd.DataFrame) -> pd.DataFrame:
    """Euclidean distance between mean window features: across domains (same class) vs across classes."""
    feats = ["amp_mean", "var1s", "cv1s"]
    g = w.groupby(["domain", "label"])[feats].mean()
    z = (g - g.mean()) / g.std()
    rows = []
    doms, labs = sorted(w.domain.unique()), sorted(w.label.unique())
    for lab in labs:
        for i, a in enumerate(doms):
            for b in doms[i + 1:]:
                if (a, lab) in z.index and (b, lab) in z.index:
                    rows.append({"kind": "domain (same class)", "dist": np.linalg.norm(z.loc[(a, lab)] - z.loc[(b, lab)])})
    for dom in doms:
        for i, a in enumerate(labs):
            for b in labs[i + 1:]:
                if (dom, a) in z.index and (dom, b) in z.index:
                    rows.append({"kind": "class (same domain)", "dist": np.linalg.norm(z.loc[(dom, a)] - z.loc[(dom, b)])})
    return pd.DataFrame(rows).groupby("kind")["dist"].describe()


# ---------------------------------------------------------------- 6.2.8 PCA / t-SNE

EMB_FEATS = ["amp_mean", "amp_std_t", "var1s", "cv1s", "amp_range", "pc1_band_0_2", "pc1_band_2_10",
             "pc1_band_10_40", "pc1_logpower", "phase_var1s"]


def plot_embeddings(name, w: pd.DataFrame, color_cols=("label", "domain")):
    x = w[EMB_FEATS].to_numpy(dtype=float, copy=True)
    for c in ["amp_mean", "amp_std_t", "var1s", "cv1s", "amp_range", "phase_var1s"]:
        j = EMB_FEATS.index(c)
        x[:, j] = np.log10(x[:, j] + 1e-9)
    x = StandardScaler().fit_transform(x)
    pca = PCA(2).fit(x)
    emb = {"PCA": pca.transform(x),
           "t-SNE": TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(x)}
    fig, axes = plt.subplots(len(color_cols), 2, figsize=(10, 4.2 * len(color_cols)))
    for i, col in enumerate(color_cols):
        cmap = LABEL_COLORS if col == "label" else (DOMAIN_COLORS if col == "domain" else
                                                     {"wallhack": "#2a78d6", "3do": "#eb6834"})
        for j, (ename, e) in enumerate(emb.items()):
            ax = axes[i, j]
            for val in sorted(w[col].unique()):
                m = (w[col] == val).to_numpy()
                ax.scatter(e[m, 0], e[m, 1], s=5, alpha=0.6, color=cmap.get(val, MUTED), label=val, lw=0)
            title = f"{ename}, color = {col}"
            if ename == "PCA":
                title += f" (expl. var {pca.explained_variance_ratio_.sum():.0%})"
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=7, markerscale=2.5)
    fig.suptitle(f"{name}: 4 s windows in simple-feature space")
    fig.tight_layout()
    save(fig, f"{name}_pca_tsne.png")


# ---------------------------------------------------------------- 6.2.9 3DO trajectory

def plot_trajectories():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    for j, day in enumerate(DAYS):
        for i, kind in enumerate(["w", "s"]):
            ax = axes[i, j]
            idx = load_index("3do")
            sc = None
            for r in idx[(idx.day == day) & (idx.session.str.startswith(kind))].itertuples():
                d = load_recording("3do", r.recording_id)
                ms = np.log10(F.motion_score(F.amplitude(d["csi"]), 100) + 1e-6)
                p = d["pos"][::10]
                sc = ax.scatter(p[:, 0], p[:, 1], c=ms[::10], s=2, cmap=SEQ, vmin=-0.5, vmax=2, lw=0)
            ax.set_title(f"{day}, {'walk' if kind == 'w' else 'sit'} recordings")
            ax.set_aspect("equal")
            if sc is not None and j == 2:
                fig.colorbar(sc, ax=ax, label="log10 rolling var (1 s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y, m")
    for ax in axes[-1]:
        ax.set_xlabel("x, m")
    fig.suptitle("3do: person position (x, y) colored by CSI motion score")
    fig.tight_layout()
    save(fig, "3do_trajectory_rollvar.png")


# ---------------------------------------------------------------- 6.3 threshold baseline

def roc_by_domain(name, w: pd.DataFrame, positives: list[str],
                  feats=("var1s", "cv1s", "var1s_clean", "cv1s_clean", "phase_var1s")):
    w = w[w.label.isin(["empty"] + positives)]
    doms = sorted(w.domain.unique())
    fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 3.8), sharey=True)
    rows = []
    for ax, feat in zip(axes, feats):
        for dom in doms:
            s = w[w.domain == dom]
            yb = (s.label != "empty").astype(int)
            fpr, tpr, _ = roc_curve(yb, s[feat])
            auc = roc_auc_score(yb, s[feat])
            rows.append({"task": name, "feature": feat, "domain": dom, "auc": auc, "n": len(s)})
            ax.plot(fpr, tpr, color=DOMAIN_COLORS[dom], lw=1.4, label=f"{dom}: AUC {auc:.3f}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_title(feat)
        ax.set_xlabel("false alarm rate")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("detection rate")
    fig.suptitle(f"Threshold detector, {name}: empty vs {' + '.join(positives)}")
    fig.tight_layout()
    save(fig, f"baseline_roc_{name}.png")
    return pd.DataFrame(rows)


def threshold_transfer(name, w: pd.DataFrame, positives: list[str], feat: str):
    """Threshold fit on domain A (max balanced accuracy), applied unchanged to domain B."""
    w = w[w.label.isin(["empty"] + positives)]
    doms = sorted(w.domain.unique())

    def fit(s):
        x, yb = np.log10(s[feat].to_numpy()), (s.label != "empty").to_numpy()
        cands = np.unique(x)
        ba = [0.5 * ((x[yb] > c).mean() + (x[~yb] <= c).mean()) for c in cands]
        return cands[int(np.argmax(ba))]

    def score(s, thr):
        x, yb = np.log10(s[feat].to_numpy()), (s.label != "empty").to_numpy()
        tpr, far = (x[yb] > thr).mean(), (x[~yb] > thr).mean()
        return 0.5 * (tpr + 1 - far), far

    ba = np.zeros((len(doms), len(doms)))
    far = np.zeros_like(ba)
    rows = []
    for i, a in enumerate(doms):
        thr = fit(w[w.domain == a])
        for j, b in enumerate(doms):
            ba[i, j], far[i, j] = score(w[w.domain == b], thr)
            rows.append({"task": name, "feature": feat, "train": a, "test": b, "threshold_log10": thr,
                         "balanced_acc": ba[i, j], "far": far[i, j]})
    fig, axes = plt.subplots(1, 2, figsize=(4.2 * 2 + 1, 3.8))
    for ax, m, title, cmap in [(axes[0], ba, "balanced accuracy", SEQ), (axes[1], far, "false alarm rate", SEQ)]:
        im = ax.imshow(m, cmap=cmap, vmin=0 if title.startswith("false") else 0.5, vmax=1 if not title.startswith("false") else max(0.5, m.max()))
        for i in range(len(doms)):
            for j in range(len(doms)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if (m[i, j] - im.norm.vmin) / (im.norm.vmax - im.norm.vmin) > 0.55 else "#0b0b0b")
        ax.set_xticks(range(len(doms)), doms, rotation=30)
        ax.set_yticks(range(len(doms)), doms)
        ax.set_xlabel("test domain")
        ax.set_ylabel("threshold fit on")
        ax.set_title(title)
        ax.grid(False)
    fig.suptitle(f"Threshold transfer, {name}, feature {feat} (diagonal = in-sample)")
    fig.tight_layout()
    save(fig, f"baseline_transfer_{name}_{feat}.png")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def run(dataset):
    w = load_windows(dataset)
    plot_dt_hist(dataset)
    plot_overview(dataset)
    steps = amp_level_steps(dataset)
    steps.to_csv(RES / f"{dataset}_amp_level_checks.csv", index=False)
    print(steps.describe().round(3).to_string())
    plot_heatmaps_and_lines(dataset)
    plot_rolling_var_time(dataset)
    plot_distributions(dataset, w)
    plot_phase(dataset)
    plot_spectra(dataset)
    plot_domain_profiles(dataset)
    print(domain_vs_class_distance(w).round(2).to_string())
    plot_embeddings(dataset, w)
    rocs, transfers = [], []
    tasks = {"wallhack": {"wallhack_motion": ["walk", "walk_wave"]},
             "3do": {"3do_motion": ["walk"], "3do_presence": ["walk", "sit", "lie"],
                     "3do_static": ["sit", "lie"]}}[dataset]
    for tname, pos in tasks.items():
        rocs.append(roc_by_domain(tname, w, pos))
        for feat in ("var1s", "cv1s_clean"):
            transfers.append(threshold_transfer(tname, w, pos, feat))
    if dataset == "3do":
        plot_trajectories()
    return pd.concat(rocs), pd.concat(transfers)


def run_cross():
    """Wallhack <-> 3DO: shared embedding and threshold transfer for the motion task."""
    wh, td = load_windows("wallhack"), load_windows("3do")
    both = pd.concat([wh.assign(domain=wh.domain), td.assign(domain="3do_" + td.domain)], ignore_index=True)
    plot_embeddings("both", both[both.label.isin(["empty", "walk"])].reset_index(drop=True),
                    color_cols=("label", "dataset"))
    m = both[both.label.isin(["empty", "walk", "walk_wave"])].copy()
    m["domain"] = m["dataset"]
    return threshold_transfer("cross_dataset_motion", m, ["walk", "walk_wave"], "cv1s_clean"), \
        threshold_transfer("cross_dataset_motion", m, ["walk", "walk_wave"], "var1s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["wallhack", "3do", "all", "cross"])
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    plot_pairing_comparison()
    rocs, transfers = [], []
    for ds in (["wallhack", "3do"] if args.dataset == "all" else ([] if args.dataset == "cross" else [args.dataset])):
        r, t = run(ds)
        rocs.append(r)
        transfers.append(t)
    if args.dataset in ("all", "cross"):
        transfers.extend(run_cross())
    if rocs:
        pd.concat(rocs).to_csv(RES / "baseline_threshold_auc.csv", index=False)
        print(pd.concat(rocs).pivot_table(index=["task", "domain"], columns="feature", values="auc").round(3).to_string())
    if transfers:
        pd.concat(transfers).to_csv(RES / "baseline_threshold_transfer.csv", index=False)


if __name__ == "__main__":
    main()
