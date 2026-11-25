"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 2: TABLAS Y FIGURAS DEL REPORTE (a partir de results/)
================================================================================
Lee los resultados que deja run_e2.py y genera, sin re-entrenar nada:
  results/e2/tables/*.tex|.md|.csv   tablas que el reporte incluye con \\input
  results/e2/figures/*.png           figuras del reporte y de la presentación
  results/e2/resumen.json            cifras clave para redactar el análisis
Así cada número del reporte sale de un archivo generado por el código.

Uso:  python analyze_e2.py [--output-root .]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn import metrics as skm

NAMES = {"mlp": "MLP", "simple": "SimpleRNN", "lstm": "LSTM"}
MLP_E2 = "MLP Tuned (re-entrenado)"   # el MLP de la E1 re-entrenado en el entorno de la E2
COLORS = {"simple": "#2a78d6", "lstm": "#eb6834", "mlp": "#1baf7a"}   # paleta validada (dataviz)
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "azul", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
PARAM_LABELS = {"seq_len": "Longitud L", "units": "Unidades", "dropout": "Dropout", "lr": "Tasa de aprendizaje"}


# ------------------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------------------

def load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(x: float | None, nd: int = 2) -> str:
    return "—" if x is None else f"{x * 100:.{nd}f}"


def num(n: int | None) -> str:
    return "—" if n is None else f"{n:,}".replace(",", "\\,")


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def write_table(out: Path, name: str, header: list[str], rows: list[list[str]], align: str,
                bold_rows: set[int] | None = None) -> None:
    """Escribe la misma tabla en LaTeX (booktabs), Markdown y CSV."""
    bold_rows = bold_rows or set()
    tex = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        cells = [f"\\textbf{{{c}}}" if i in bold_rows and c not in ("", "—") else c for c in r]
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    def plain(s: str) -> str:
        return (s.replace("\\,", " ").replace("$\\pm$", "±").replace("$\\checkmark$", "✓")
                .replace("\\%", "%").replace("\\_", "_").replace("$\\le$", "≤").replace("$>$", ">"))

    md = ["| " + " | ".join(plain(h) for h in header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(plain(c) for c in r) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([plain(h) for h in header])
        w.writerows([[plain(c) for c in r] for r in rows])


# ------------------------------------------------------------------------------
# Tablas
# ------------------------------------------------------------------------------

def tables_eda(eda: dict, out: Path) -> None:
    names = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
    rows = []
    for s in ("train", "val", "test"):
        d = eda["longitud_tokens"][s]
        rows.append([names[s], num(eda["documentos"][s]), f"{d['media']:.0f}", f"{d['mediana']:.0f}",
                     f"{d['p75']:.0f}", f"{d['p90']:.0f}", num(d["maximo"]), str(eda["vacios"][s])])
    write_table(out, "eda_longitudes", ["Partición", "Documentos", "Media", "Mediana", "p75", "p90", "Máximo", "Vacíos"],
                rows, "lrrrrrrr")
    rows = [[f"L = {L}", f"{v:.1f}"] for L, v in eda["pct_completos_train"].items()]
    write_table(out, "eda_L", ["Longitud", "\\% de documentos sin truncar"], rows, "lr")
    rows = [[f"V = {int(V):,}".replace(",", "\\,"), f"{c * 100:.1f}"] for V, c in eda["cobertura_val"].items()]
    write_table(out, "eda_V", ["Vocabulario", "\\% de palabras de validación cubiertas"], rows, "lr")


def table_tuning(cell: str, tun: dict, out: Path) -> None:
    rows, bold = [], set()
    for k, step in enumerate(tun["pasos"], start=1):
        for t in step["pruebas"]:
            chosen = t["valor"] == step["elegido"]
            if chosen:
                bold.add(len(rows))
            rows.append([str(k), PARAM_LABELS[step["parametro"]], f"{t['valor']:g}", pct(t["val_f1_macro"]),
                         pct(t["val_accuracy"]), str(t["mejor_epoca"]), f"{t['s_por_epoca']:.1f}",
                         "$\\checkmark$" if chosen else ""])
    write_table(out, f"ajuste_{cell}", ["Paso", "Parámetro", "Valor", "F1 macro val. (\\%)", "Exactitud val. (\\%)",
                                        "Mejor época", "s/época", "Elegido"], rows, "clrrrrrc", bold)


def table_base_vs_tuned(tunings: dict, out: Path) -> None:
    rows = []
    for cell, tun in tunings.items():
        for label, cfg, val in (("base", tun["base_config"], tun["base_val"]),
                                ("ajustado", tun["config_final"], tun["final_val"])):
            rows.append([f"{NAMES[cell]} ({label})", str(cfg["seq_len"]), str(cfg["units"]), f"{cfg['dropout']:g}",
                         f"{cfg['lr']:g}", pct(val["f1_macro"]), pct(val["accuracy"])])
    write_table(out, "base_vs_ajustado", ["Modelo", "L", "Unidades", "Dropout", "lr", "F1 macro val. (\\%)",
                                          "Exactitud val. (\\%)"], rows, "lrrrrrr")


def table_final(root: Path, finals: dict, mlp_e2: dict | None, out: Path) -> list[dict]:
    rows_data = []
    for key, label in (("baseline_mlp", "MLP Baseline (E1)"),
                       ("tuned_mlp", "MLP Tuned (E1)")):
        r = load(root / "results" / f"{key}.json")
        if r:
            rows_data.append({"modelo": label, "params": None, "test": r, "best": None, "spe": None})
    if mlp_e2:
        rows_data.append({"modelo": MLP_E2, "params": mlp_e2.get("params"),
                          "test": mlp_e2, "best": mlp_e2.get("best_epoch"), "spe": mlp_e2.get("sec_per_epoch")})
    for cell in ("simple", "lstm"):
        f = finals.get(cell)
        if f:
            rows_data.append({"modelo": f"{NAMES[cell]} (ajustada)", "params": f["params"]["total"], "test": f["test"],
                              "best": f["best_epoch"], "spe": f["sec_per_epoch"]})
    rows = [[d["modelo"], num(d["params"]), pct(d["test"]["accuracy"]), pct(d["test"]["f1_macro"]),
             pct(d["test"]["precision_macro"]), pct(d["test"]["recall_macro"]),
             "—" if d["best"] is None else str(d["best"]), "—" if d["spe"] is None else f"{d['spe']:.1f}"]
            for d in rows_data]
    # Métricas en %: el pie de la tabla lo indica (encabezados cortos para que la tabla quepa sin reducirla)
    write_table(out, "final", ["Modelo", "Parámetros", "Exactitud", "F1 macro", "Precisión",
                               "Exhaustividad", "Mejor época", "s/época"], rows, "lrrrrrrr")
    return rows_data


def table_seeds(root: Path, out: Path) -> dict:
    res = {}
    rows = []
    for key, label in (("mlp", "MLP Tuned (re-entrenado)"), ("simple", "SimpleRNN (ajustada)"),
                       ("lstm", "LSTM (ajustada)")):
        s = load(root / "results" / "e2" / "final" / f"{key}_semillas.json")
        if s:
            res[key] = s
            seeds = ", ".join(str(r["seed"]) for r in s["por_semilla"])
            rows.append([label, f"{s['accuracy']['media'] * 100:.2f} $\\pm$ {s['accuracy']['desv'] * 100:.2f}",
                         f"{s['f1_macro']['media'] * 100:.2f} $\\pm$ {s['f1_macro']['desv'] * 100:.2f}", seeds])
    if rows:
        write_table(out, "semillas", ["Modelo", "Exactitud (\\%)", "F1 macro (\\%)", "Semillas"], rows, "lrrl")
    return res


def table_por_semilla(root: Path, out: Path, finals: dict) -> dict:
    """F1 macro de cada semilla, en validación y en prueba, con la configuración final.

    El modelo final de cada RNN es la semilla con mejor F1 macro de VALIDACIÓN (la prueba no
    interviene en la elección); su fila va en negrita. Se reportan también las demás semillas.
    """
    e2 = root / "results" / "e2"
    log_path = e2 / "bitacora.csv"
    if len(finals) < 2 or not log_path.exists():
        return {}
    with open(log_path, encoding="utf-8") as f:
        log = {r["run_id"]: r for r in csv.DictReader(f)}
    per = {}
    for cell in ("simple", "lstm"):
        s = load(e2 / "final" / f"{cell}_semillas.json")
        if not s:
            return {}
        base = finals[cell]["run_id"].rsplit("_s", 1)[0]
        per[cell] = {}
        for r in s["por_semilla"]:
            run = log.get(f"{base}_s{r['seed']}")
            val = float(run["val_f1_macro"]) if run and run.get("val_f1_macro") else None
            per[cell][r["seed"]] = (val, r["f1_macro"])
    mlp = load(e2 / "final" / "mlp_semillas.json")
    mlp_test = {r["seed"]: r["f1_macro"] for r in mlp["por_semilla"]} if mlp else {}
    seeds = list(per["simple"])
    chosen = {c: max(per[c], key=lambda k: per[c][k][0] if per[c][k][0] is not None else -1) for c in per}

    def pct(v):
        return "—" if v is None else f"{v * 100:.2f}"

    def mean_sd(vals):
        v = np.array([x for x in vals if x is not None]) * 100
        return f"{v.mean():.2f} $\\pm$ {v.std(ddof=1):.2f}" if len(v) > 1 else "—"

    rows, bold = [], set()
    for i, sd in enumerate(seeds):
        marks = [NAMES[c] for c in per if chosen[c] == sd]
        label = f"{sd} (final)" if marks else str(sd)
        if len(marks) == len(per):
            bold.add(i)
        rows.append([label, pct(per["simple"][sd][0]), pct(per["simple"][sd][1]),
                     pct(per["lstm"][sd][0]), pct(per["lstm"][sd][1]), pct(mlp_test.get(sd))])
    rows.append(["Media $\\pm$ d.e.",
                 mean_sd([per["simple"][k][0] for k in seeds]), mean_sd([per["simple"][k][1] for k in seeds]),
                 mean_sd([per["lstm"][k][0] for k in seeds]), mean_sd([per["lstm"][k][1] for k in seeds]),
                 mean_sd([mlp_test.get(k) for k in seeds])])
    write_table(out, "por_semilla", ["Semilla", "SimpleRNN val.", "SimpleRNN prueba", "LSTM val.",
                                     "LSTM prueba", "MLP prueba"], rows, "lrrrrr", bold_rows=bold)
    return {"semilla_final": chosen, "por_semilla": {c: {str(k): v for k, v in d.items()} for c, d in per.items()}}


# ------------------------------------------------------------------------------
# Figuras
# ------------------------------------------------------------------------------

def fig_validation(root: Path, finals: dict, figs: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cell, f in finals.items():
        h = load(root / "results" / "e2" / "runs" / f["run_id"] / "history.json")
        ep = np.arange(1, len(h["val_loss"]) + 1)
        for ax, key in zip(axes, ("val_loss", "val_accuracy")):
            ax.plot(ep, h[key], color=COLORS[cell], lw=2, marker="o", ms=4, label=NAMES[cell])
            b = f["best_epoch"]
            ax.plot([b], [h[key][b - 1]], "o", color=COLORS[cell], ms=9, mec="white", mew=1.5)
    axes[0].set_ylabel("Pérdida de validación")
    axes[1].set_ylabel("Exactitud de validación")
    for ax in axes:
        ax.set_xlabel("Época")
        style(ax)
        ax.legend(frameon=False)
    fig.suptitle("Validación por época (el punto grande marca la mejor época)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "comparacion_validacion.png", dpi=200)
    plt.close(fig)


def fig_tuning(tunings: dict, figs: Path) -> None:
    params = [s["parametro"] for s in next(iter(tunings.values()))["pasos"]]
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, param in zip(axes, params):
        for cell, tun in tunings.items():
            step = next(s for s in tun["pasos"] if s["parametro"] == param)
            xs = [f"{t['valor']:g}" for t in step["pruebas"]]
            ys = [t["val_f1_macro"] * 100 for t in step["pruebas"]]
            ax.plot(xs, ys, color=COLORS[cell], lw=2, marker="o", ms=5, label=NAMES[cell])
            k = [t["valor"] for t in step["pruebas"]].index(step["elegido"])
            ax.plot([xs[k]], [ys[k]], "o", color=COLORS[cell], ms=10, mec="white", mew=1.5)
        ax.set_title(PARAM_LABELS[param], fontsize=10)
        style(ax)
    axes[0].set_ylabel("F1 macro de validación (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Ajuste uno a la vez (el punto grande es el valor elegido)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "ajuste.png", dpi=200)
    plt.close(fig)


def fig_confusion(cm: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if norm[i, j] >= 0.10:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Proporción de la clase real")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def length_analysis(root: Path, eda: dict, finals: dict, out: Path, figs: Path) -> dict | None:
    preds = root / "results" / "e2" / "preds"
    if not (preds / "test_lengths.npy").exists() or not (preds / "y_test.npy").exists():
        return None
    lengths = np.load(preds / "test_lengths.npy")
    y = np.load(preds / "y_test.npy")
    tr_len = eda["longitud_tokens"]["train"]
    cuts = [tr_len["p25"], tr_len["mediana"], tr_len["p75"]]
    edges = [-1] + cuts + [np.inf]
    labels = [f"$\\le$ {cuts[0]:.0f}", f"{cuts[0]:.0f}–{cuts[1]:.0f}", f"{cuts[1]:.0f}–{cuts[2]:.0f}",
              f"$>$ {cuts[2]:.0f}"]
    models = {}
    for key in ("mlp", "simple", "lstm"):
        f = preds / f"{key}_test_probs.npy"
        if f.exists():
            models[key] = np.load(f).argmax(axis=1)
    bins = np.digitize(lengths, edges[1:-1], right=True)
    # Una sola columna de truncados si las dos RNN usan la misma L
    Ls = {cell: finals[cell]["config"]["seq_len"] if cell in finals else None for cell in ("simple", "lstm")}
    trunc_L = [(None, Ls["simple"])] if Ls["simple"] == Ls["lstm"] else list(Ls.items())
    rows, acc = [], {k: [] for k in models}
    for b, lab in enumerate(labels):
        m = bins == b
        row = [lab + " tokens", str(int(m.sum()))]
        for cell, L in trunc_L:
            row.append("—" if L is None else f"{np.mean(lengths[m] > L) * 100:.0f}")
        for k, pred in models.items():
            a = float(np.mean(pred[m] == y[m])) if m.any() else float("nan")
            acc[k].append(a)
            row.append(pct(a, 1))
        rows.append(row)
    # Exactitud en %: el pie de la tabla lo indica
    header = ["Longitud", "Documentos"] + \
             [f"Truncados, L = {L} (\\%)" if cell is None else f"Truncados {NAMES[cell]} (\\%)"
              for cell, L in trunc_L] + [NAMES[k] for k in models]
    write_table(out, "longitud", header, rows, "l" + "r" * (len(header) - 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    w = 0.8 / max(len(models), 1)
    for i, (k, vals) in enumerate(acc.items()):
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, np.array(vals) * 100, w * 0.92,
                      color=COLORS[k], label=NAMES[k])
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v * 100:.0f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("$\\le$", "≤").replace("$>$", ">") + " tokens" for l in labels])
    ax.set_ylabel("Exactitud en test (%)")
    ax.set_title("Exactitud según la longitud del documento (cuartiles de train)", fontsize=11)
    style(ax)
    ax.legend(frameon=False, ncol=len(models))
    fig.tight_layout()
    fig.savefig(figs / "longitud.png", dpi=200)
    plt.close(fig)
    return {"cortes": cuts, "exactitud": acc}


def fig_final_bars(rows_data: list[dict], figs: Path) -> None:
    keys = [(MLP_E2, "mlp"), ("SimpleRNN (ajustada)", "simple"),
            ("LSTM (ajustada)", "lstm")]
    present = [(d, k) for name, k in keys for d in rows_data if d["modelo"] == name]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics_ = [("accuracy", "Exactitud"), ("f1_macro", "F1 macro")]
    x = np.arange(len(metrics_))
    w = 0.8 / len(present)
    for i, (d, k) in enumerate(present):
        vals = [d["test"][m] * 100 for m, _ in metrics_]
        bars = ax.bar(x + (i - (len(present) - 1) / 2) * w, vals, w * 0.92, color=COLORS[k], label=NAMES[k])
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics_])
    ax.set_ylabel("Test (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Resultado en test (semilla 42)", fontsize=11)
    style(ax)
    ax.legend(frameon=False, ncol=len(present))
    fig.tight_layout()
    fig.savefig(figs / "comparacion_final.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Principal
# ------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Tablas y figuras de la Entrega 2")
    p.add_argument("--output-root", default=".")
    a = p.parse_args(argv)
    root = Path(a.output_root)
    e2 = root / "results" / "e2"
    out, figs = e2 / "tables", e2 / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    eda = load(e2 / "eda.json")
    tunings = {c: t for c in ("simple", "lstm") if (t := load(e2 / f"ajuste_{c}.json"))}
    finals = {c: f for c in ("simple", "lstm") if (f := load(e2 / "final" / f"{c}.json"))}
    mlp_e2 = load(root / "results" / "tuned_mlp_e2.json")
    class_names = load(root / "models" / "e2" / "class_names.json")

    if eda:
        tables_eda(eda, out)
    for cell, tun in tunings.items():
        table_tuning(cell, tun, out)
    if tunings:
        table_base_vs_tuned(tunings, out)
        fig_tuning(tunings, figs)
    rows_data = table_final(root, finals, mlp_e2, out)
    fig_final_bars(rows_data, figs)
    seeds = table_seeds(root, out)
    seed_choice = table_por_semilla(root, out, finals)
    if len(finals) == 2:
        fig_validation(root, finals, figs)

    # Matriz de confusión del mejor modelo recurrente (por F1 macro en test) y F1 por clase
    summary = {"finales": {}, "semillas": seeds, "eleccion_semilla": seed_choice}
    if finals and class_names:
        best = max(finals, key=lambda c: finals[c]["test"]["f1_macro"])
        cm = np.load(e2 / "preds" / f"{best}_confusion.npy")
        fig_confusion(cm, class_names, f"Matriz de confusión normalizada: {NAMES[best]} (test)",
                      figs / "confusion_mejor.png")
        y = np.load(e2 / "preds" / "y_test.npy")
        f1s = {}
        for key in ("mlp", "simple", "lstm"):
            f = e2 / "preds" / f"{key}_test_probs.npy"
            if f.exists():
                f1s[key] = skm.f1_score(y, np.load(f).argmax(axis=1), average=None,
                                        labels=list(range(len(class_names))), zero_division=0)
        rows = [[c.replace("_", "\\_")] + [pct(float(f1s[k][i]), 1) for k in f1s] for i, c in enumerate(class_names)]
        write_table(out, "por_clase", ["Clase"] + [f"F1 {NAMES[k]} (\\%)" for k in f1s], rows, "l" + "r" * len(f1s))
        # Pares más confundidos del mejor modelo
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        off = [(norm[i, j], class_names[i], class_names[j]) for i in range(len(cm)) for j in range(len(cm)) if i != j]
        summary["mejor_rnn"] = best
        summary["confusiones_principales"] = [{"real": r, "predicha": p_, "proporcion": float(v)}
                                              for v, r, p_ in sorted(off, reverse=True)[:6]]
        summary["f1_por_clase"] = {NAMES[k]: {c: float(v[i]) for i, c in enumerate(class_names)} for k, v in f1s.items()}

    if eda and finals:
        summary["longitud"] = length_analysis(root, eda, finals, out, figs)
    for cell, f in finals.items():
        summary["finales"][cell] = {k: f[k] for k in ("config", "params", "epochs_run", "best_epoch",
                                                      "sec_per_epoch", "train_time_sec", "val", "test")}
    if mlp_e2:
        summary["mlp_e2"] = {k: mlp_e2.get(k) for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro",
                                                        "params", "best_epoch", "epochs_run", "sec_per_epoch")}
    if eda:
        summary["eda"] = {"longitud_train": eda["longitud_tokens"]["train"], "vacios": eda["vacios"],
                          "pct_completos_train": eda["pct_completos_train"], "cobertura_val": eda["cobertura_val"],
                          "razon_S_W": eda["razon_S_W"], "vocabulario_train_total": eda["vocabulario_train_total"]}
    with open(e2 / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"Tablas en {out} | Figuras en {figs} | Resumen en {e2 / 'resumen.json'}")
    return summary


if __name__ == "__main__":
    main()
