"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 2: EDA MÍNIMA DE SECUENCIAS (justifica L y V)
================================================================================
Mide sobre los mismos datos de la E1:
  - longitud de los documentos en tokens (con la misma estandarización que TextVectorization)
  - % de documentos que se truncarían con cada L candidato
  - cobertura del vocabulario de train sobre validación para cada V candidato
  - documentos vacíos tras quitar encabezados, firmas y citas
Salidas: results/e2/eda.json, results/e2/figures/eda_longitudes_vocabulario.png
         y results/e2/preds/test_lengths.npy (para el análisis por longitud).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras

from data_utils import load_20newsgroups
from train_rnn import results_dir, set_output_root, to_tensor

L_CANDIDATES = (100, 200, 300, 400)
V_CANDIDATES = (10_000, 20_000, 30_000)


def describe(lengths: np.ndarray) -> dict:
    return {
        "media": float(lengths.mean()),
        "mediana": float(np.median(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "maximo": int(lengths.max()),
    }


def main(argv: list[str] | None = None, data: dict | None = None) -> dict:
    p = argparse.ArgumentParser(description="EDA de secuencias para la Entrega 2")
    p.add_argument("--output-root", default=".")
    a = p.parse_args(argv)
    set_output_root(a.output_root)
    out = results_dir()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(parents=True, exist_ok=True)

    data = data or load_20newsgroups()
    texts = {s: data[s][0] for s in ("train", "val", "test")}

    # Vectorizador SIN límite de vocabulario, ajustado solo con train (mismo criterio que el modelo)
    vec = keras.layers.TextVectorization(output_mode="int", ragged=True)
    vec.adapt(to_tensor(texts["train"]))
    vocab = vec.get_vocabulary()                       # ordenado por frecuencia: '', [UNK], más frecuente, ...
    ids = {s: vec(to_tensor(texts[s])) for s in texts}
    lengths = {s: ids[s].row_lengths().numpy() for s in texts}

    tr = lengths["train"]
    val_ids = ids["val"].flat_values.numpy()
    coverage = {str(V): float(np.mean((val_ids > 1) & (val_ids < V))) for V in V_CANDIDATES}
    words_mean, words_median = float(tr.mean()), float(np.median(tr))

    eda = {
        "documentos": {s: len(texts[s]) for s in texts},
        "longitud_tokens": {s: describe(lengths[s]) for s in texts},
        "pct_truncados_train": {str(L): float(np.mean(tr > L) * 100) for L in L_CANDIDATES},
        "pct_completos_train": {str(L): float(np.mean(tr <= L) * 100) for L in L_CANDIDATES},
        "vocabulario_train_total": len(vocab) - 2,
        "cobertura_val": coverage,
        "vacios": {s: int((lengths[s] == 0).sum()) for s in texts},
        "razon_S_W": {"S": len(texts["train"]), "W_media": words_mean, "W_mediana": words_median,
                      "S_sobre_W_media": len(texts["train"]) / words_mean,
                      "S_sobre_W_mediana": len(texts["train"]) / words_median},
        "clases": {
            s: {name: int((data[s][1] == i).sum()) for i, name in enumerate(data["class_names"])}
            for s in ("train", "val", "test")
        },
    }
    with open(out / "eda.json", "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2, ensure_ascii=False)
    np.save(out / "preds" / "test_lengths.npy", lengths["test"].astype("int32"))

    # ---- Figura: (a) histograma de longitudes con los L candidatos; (b) cobertura según V ----
    ink, muted, grid = "#0b0b0b", "#52514e", "#e1e0d9"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    nz = tr[tr > 0]
    bins = np.logspace(0, np.log10(nz.max() + 1), 50)
    ax1.hist(nz, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.5)
    ax1.set_xscale("log")
    for k, L in enumerate((100, 200, 400)):
        ax1.axvline(L, color=muted, lw=1, ls="--")
        ax1.annotate(f"L={L}: {eda['pct_completos_train'][str(L)]:.0f} % completos", xy=(L, 0.97 - 0.09 * k),
                     xycoords=("data", "axes fraction"), xytext=(3, 0), textcoords="offset points",
                     fontsize=8, color=muted, va="top")
    ax1.set_xlabel("Tokens por documento (escala log)")
    ax1.set_ylabel("Documentos de entrenamiento")
    ax1.set_title("Longitud de los documentos", fontsize=10, color=ink)

    Vs = np.unique(np.linspace(1_000, min(50_000, len(vocab)), 60).astype(int))
    cov = [np.mean((val_ids > 1) & (val_ids < V)) * 100 for V in Vs]
    ax2.plot(Vs, cov, color="#2a78d6", lw=2)
    for V in V_CANDIDATES:
        if V <= len(vocab):
            c = coverage[str(V)] * 100
            ax2.plot([V], [c], "o", color="#2a78d6", ms=7, mec="white", mew=1.5)
            ax2.annotate(f"{V // 1000} mil: {c:.1f} %", xy=(V, c), xytext=(4, -12),
                         textcoords="offset points", fontsize=8, color=muted)
    ax2.set_xlabel("Tamaño del vocabulario V")
    ax2.set_ylabel("% de palabras de validación cubiertas")
    ax2.set_title("Cobertura del vocabulario", fontsize=10, color=ink)
    for ax in (ax1, ax2):
        ax.grid(True, color=grid, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(out / "figures" / "eda_longitudes_vocabulario.png", dpi=200)
    plt.close(fig)

    d = eda["longitud_tokens"]["train"]
    print("\n--- EDA DE SECUENCIAS ---")
    print(f"Documentos: {eda['documentos']}")
    print(f"Longitud en train: media {d['media']:.0f}, mediana {d['mediana']:.0f}, p75 {d['p75']:.0f}, "
          f"p90 {d['p90']:.0f}, máx {d['maximo']}")
    print(f"% sin truncar: {', '.join(f'L={k}: {v:.1f} %' for k, v in eda['pct_completos_train'].items())}")
    print(f"Cobertura en validación: {', '.join(f'V={k}: {v * 100:.1f} %' for k, v in coverage.items())}")
    print(f"Vacíos: {eda['vacios']} | S/W (media) = {eda['razon_S_W']['S_sobre_W_media']:.1f}")
    return eda


if __name__ == "__main__":
    main()
