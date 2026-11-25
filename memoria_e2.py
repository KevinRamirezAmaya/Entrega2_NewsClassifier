"""
Entrega 2 - Prueba de memoria con distancia controlada.

Cada frase de prueba tiene un tema claro (su clase esperada) y va AL INICIO del texto;
después se agregan k palabras de un texto de relleno sin tema (se usan tres rellenos
distintos para que el resultado no dependa de uno solo). Como las RNN clasifican con su último
estado, la información del tema tiene que atravesar k pasos para llegar a la salida.
Se mide, para cada k, la probabilidad media de la clase esperada y el % de frases en que
esa clase sigue siendo la primera opción. El MLP (TF-IDF) no depende del orden: sirve de
referencia.

Usa los modelos guardados en models/e2 (no descarga datos). Salidas:
  results/e2/memoria.json
  results/e2/figures/memoria.png
  results/e2/tables/memoria.{csv,md,tex}

Uso:  python memoria_e2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import demo_e2 as demo
import train_rnn as tr

# Frases con un tema claro y la clase que les corresponde
FRASES = [
    ("NASA rover detects evidence of liquid water and atmosphere on Mars.", "sci.space"),
    ("The engine needs a new oil filter and spark plugs for better mileage.", "rec.autos"),
    (demo.KEY_SENTENCE.replace("Anyway, t", "T"), "rec.sport.hockey"),
    ("The doctor prescribed antibiotics because the patient had a bacterial infection.", "sci.med"),
    ("For sale: used laptop in good condition, asking 300 dollars plus shipping.", "misc.forsale"),
    ("The pitcher threw a perfect game and the batter hit a home run in the ninth inning.", "rec.sport.baseball"),
    ("Israeli and Palestinian leaders met to discuss the peace process in the region.", "talk.politics.mideast"),
    ("I need software to render 3D images and convert them to GIF and JPEG formats.", "comp.graphics"),
]
# Palabras de relleno agregadas DESPUÉS de la frase (todas caben en L = 400)
DISTANCIAS = [0, 10, 25, 50, 100, 150, 200, 300]
RELLENOS = [
    demo.FILLER.strip(),
    "Please let me know if you have any questions or comments about this and I will try to answer as soon as possible.",
    "Yesterday we went out for a long walk with some friends and then we talked for hours about many different things.",
]


def texto(frase: str, relleno: str, k: int) -> str:
    palabras = relleno.split()
    return " ".join([frase] + (palabras * (k // len(palabras) + 1))[:k])


def main() -> None:
    class_names = json.loads((demo.MODELS / "class_names.json").read_text(encoding="utf-8"))
    modelos = [m for m in (demo.load_mlp(class_names), demo.load_rnn("simple", class_names),
                           demo.load_rnn("lstm", class_names)) if m is not None]
    esperadas = [class_names.index(c) for _, c in FRASES]

    res = {"frases": [{"texto": f, "clase": c} for f, c in FRASES], "distancias": DISTANCIAS,
           "rellenos": RELLENOS, "casos_por_distancia": len(FRASES) * len(RELLENOS), "modelos": {}}
    y = np.array(esperadas * len(RELLENOS))
    for m in modelos:
        prob, acierto, por_relleno = [], [], []
        for k in DISTANCIAS:
            p = m.predict_fn([texto(f, r, k) for r in RELLENOS for f, _ in FRASES])
            ok = p.argmax(axis=1) == y
            prob.append(float(np.mean(p[np.arange(len(y)), y])))
            acierto.append(float(np.mean(ok)))
            por_relleno.append([float(np.mean(o)) for o in np.split(ok, len(RELLENOS))])
        res["modelos"][m.name] = {"prob_media": prob, "aciertos": acierto,
                                  "aciertos_por_relleno": por_relleno}

    out = tr.results_dir()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "memoria.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # Tabla: % de frases con la clase esperada como primera opción
    nombres = list(res["modelos"])
    filas = [[str(k)] + [f"{res['modelos'][n]['aciertos'][i] * 100:.0f}" for n in nombres]
             for i, k in enumerate(DISTANCIAS)]
    cab = ["Palabras después de la frase"] + [f"{n.replace(' (E1)', '')}: aciertos (%)" for n in nombres]
    (out / "tables" / "memoria.csv").write_text(
        "\n".join(",".join(r) for r in [cab] + filas) + "\n", encoding="utf-8")
    (out / "tables" / "memoria.md").write_text(
        "| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n"
        + "".join("| " + " | ".join(r) + " |\n" for r in filas), encoding="utf-8")
    tex = ["\\begin{tabular}{" + "r" * len(cab) + "}", "\\toprule",
           "\\textbf{Palabras después} & " + " & ".join(
               f"\\textbf{{{n.replace(' (E1)', '')}}}" for n in nombres) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in filas] + ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "memoria.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # Figura: probabilidad media de la clase esperada según la distancia
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colores = {"MLP (E1)": tr.COLORS.get("mlp", "#1baf7a"), "SimpleRNN": tr.COLORS["simple"],
               "LSTM": tr.COLORS["lstm"]}
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    for n in nombres:
        y = np.array(res["modelos"][n]["prob_media"]) * 100
        ax.plot(DISTANCIAS, y, marker="o", markersize=5, linewidth=2,
                color=colores.get(n, "#555555"), label=n.replace(" (E1)", ""))
        ax.annotate(f"{y[-1]:.0f}", (DISTANCIAS[-1], y[-1]), textcoords="offset points",
                    xytext=(6, -3), fontsize=8, color="#444444")
    ax.axhline(100 / len(class_names), color="#888888", linewidth=1, linestyle=":",
               label=f"azar (1 de {len(class_names)})")
    ax.set_xlabel("Palabras de relleno después de la frase con el tema")
    ax.set_ylabel("Probabilidad media de la clase correcta (%)")
    ax.set_title(f"Prueba de memoria: {len(FRASES)} frases × {len(RELLENOS)} rellenos, la frase va al inicio")
    ax.set_ylim(0, 100)
    ax.set_xticks(DISTANCIAS)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "memoria.png")
    plt.close(fig)

    print("Probabilidad media de la clase correcta (%) y aciertos (%) según la distancia")
    print("k".rjust(5) + "".join(f"{n:>24s}" for n in nombres))
    for i, k in enumerate(DISTANCIAS):
        print(f"{k:5d}" + "".join(
            f"{res['modelos'][n]['prob_media'][i] * 100:14.1f} / {res['modelos'][n]['aciertos'][i] * 100:5.0f}"
            for n in nombres))
    print(f"\nGuardado en {out / 'memoria.json'}, {out / 'figures' / 'memoria.png'}")


if __name__ == "__main__":
    main()
