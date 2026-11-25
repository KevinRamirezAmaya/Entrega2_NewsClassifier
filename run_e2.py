"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 2: EJECUCIÓN COMPLETA (EDA -> AJUSTE -> MODELOS FINALES -> REFERENCIA MLP)
================================================================================
Pasos (PLAN_ENTREGA2.md §3, §6 y §7):
  1. EDA de secuencias (longitudes, vocabulario, vacíos).
  2. Configuración base de SimpleRNN y LSTM (mismos hiperparámetros).
  3. Ajuste de parámetros UNO A LA VEZ (L, unidades, dropout, lr), eligiendo por F1 macro de
     validación. Un cambio se adopta solo si mejora más de 0.5 puntos.
  4. Modelos finales: la mejor configuración de cada uno se evalúa UNA vez en test.
  5. Semillas 1, 7 y 123 de la E1 con la configuración final (media ± desviación).
  6. MLP Tuned de la E1 re-entrenado en el mismo entorno (referencia, análisis y demo).
Si se interrumpe, al volver a ejecutarlo salta todo lo que ya terminó.

Uso:  python run_e2.py                 (ejecución completa)
      python run_e2.py --quick-test    (prueba de humo de pocos minutos, en ./prueba_rapida)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras

import eda_sequences
import train_rnn as tr
from data_utils import load_20newsgroups

CELLS = ["simple", "lstm"]
BASE = {k: v for k, v in tr.DEFAULTS.items() if k != "cell"}
SEARCH = [                       # orden del PLAN §6
    ("seq_len", [100, 200, 400]),
    ("units", [64, 128, 256]),
    ("dropout", [0.2, 0.3, 0.5]),
    ("lr", [1e-3, 5e-4]),
]
# Por debajo de 0.5 puntos de F1 macro la diferencia se trata como ruido: en la E1 la exactitud
# varió entre ±0.3 y ±0.9 puntos solo por cambiar la semilla.
MIN_GAIN = 0.005
EXTRA_SEEDS = [1, 7, 123]        # mismas semillas de la E1 (42 ya es la principal)
MLP_TUNED_ARGS = ["--vocab-size", "10000", "--hidden-dim", "512", "--dropout", "0.3",
                  "--l2-reg", "0.00005", "--lr", "0.0005", "--epochs", "25"]   # E1, README paso 3


class Tee(io.TextIOBase):
    """Escribe en pantalla y en results/e2/log_ejecucion.txt al mismo tiempo."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def working_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_or_load(cfg: dict, seq: tr.SequenceData, class_names: list[str], phase: str,
                eval_test: bool = False) -> dict:
    """Entrena la configuración o, si ya existe (mismo run_id), reutiliza su resultado."""
    rid = tr.run_id(cfg)
    mfile = tr.runs_dir() / rid / "metrics.json"
    ckpt = tr.checkpoints_dir() / f"{rid}.weights.h5"
    if mfile.exists() and ckpt.exists():
        with open(mfile, encoding="utf-8") as f:
            res = json.load(f)
        if not eval_test or "test" in res:
            print(f"(ya existe) {rid}: val_F1={res['val']['f1_macro']:.4f}")
            return res
    return tr.train_and_evaluate(cfg, seq, class_names, phase=phase, eval_test=eval_test)


def tune(cell: str, seq: tr.SequenceData, class_names: list[str], base: dict, search: list) -> dict:
    """Ajuste uno a la vez: cada valor final queda con su propia evidencia (F1 macro de validación)."""
    out = tr.results_dir() / f"ajuste_{cell}.json"
    current = dict(base, cell=cell)
    base_res = run_or_load(current, seq, class_names, phase="base")
    log = {"cell": cell, "min_gain": MIN_GAIN, "base_config": dict(current),
           "base_run": base_res["run_id"], "base_val": base_res["val"], "pasos": []}

    for param, values in search:
        trials = []
        for v in values:
            cfg = dict(current, **{param: v})
            res = run_or_load(cfg, seq, class_names, phase=f"ajuste_{param}")
            trials.append({"valor": v, "run_id": res["run_id"], "val_f1_macro": res["val"]["f1_macro"],
                           "val_accuracy": res["val"]["accuracy"], "val_loss": res["val"]["loss"],
                           "mejor_epoca": res["best_epoch"], "s_por_epoca": res["sec_per_epoch"]})
        score = {t["valor"]: t["val_f1_macro"] for t in trials}
        previous = current[param]
        best = max(score, key=score.get)
        gain = score[best] - score[previous]
        adopted = best != previous and gain > MIN_GAIN
        if adopted:
            current[param] = best
        reason = (f"{best} mejora el F1 macro de validación en {gain * 100:.2f} puntos" if adopted else
                  f"ningún valor mejora más de {MIN_GAIN * 100:.1f} puntos; se mantiene {previous}")
        log["pasos"].append({"parametro": param, "pruebas": trials, "anterior": previous,
                             "elegido": current[param], "ganancia": gain, "adoptado": adopted, "razon": reason})
        print(f"[{tr.CELL_NAMES[cell]}] {param}: {reason}")

    final_res = run_or_load(current, seq, class_names, phase="final")
    log["config_final"] = current
    log["final_run"] = final_res["run_id"]
    log["final_val"] = final_res["val"]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return current


def finalize(cell: str, cfg: dict, seq: tr.SequenceData, class_names: list[str]) -> dict:
    """Evalúa UNA vez en test el modelo elegido (los mismos pesos que se vieron en validación) y lo guarda."""
    final_dir = tr.results_dir() / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    out = final_dir / f"{cell}.json"
    if out.exists():
        with open(out, encoding="utf-8") as f:
            return json.load(f)

    rid = tr.run_id(cfg)
    with open(tr.runs_dir() / rid / "metrics.json", encoding="utf-8") as f:
        run = json.load(f)
    keras.backend.clear_session()
    model = tr.build_rnn(cell, seq.n_tokens, cfg["embed_dim"], cfg["units"], cfg["dropout"],
                         cfg["lr"], cfg["clipnorm"], num_classes=len(class_names))
    model.load_weights(tr.checkpoints_dir() / f"{rid}.weights.h5")

    X_te, y_te = seq.padded("test", cfg["seq_len"]), seq.labels["test"]
    test_metrics, probs = tr.evaluate(model, X_te, y_te)
    y_pred = probs.argmax(axis=1)
    from sklearn import metrics as skm
    per_class = skm.classification_report(y_te, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = skm.confusion_matrix(y_te, y_pred, labels=list(range(len(class_names))))

    preds = tr.results_dir() / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    np.save(preds / f"{cell}_test_probs.npy", probs.astype("float32"))
    np.save(preds / "y_test.npy", y_te.astype("int32"))
    np.save(preds / f"{cell}_confusion.npy", cm)

    # Artefactos para la demo (sin internet): pesos, configuración, vocabulario y clases.
    # Se guardan desde el modelo recién cargado: solo pesos de la red, sin el estado del optimizador.
    md = tr.models_dir()
    md.mkdir(parents=True, exist_ok=True)
    model.save_weights(md / f"{cell}_final.weights.h5")
    with open(md / f"{cell}_final_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    vocab_file = md / f"vocab_V{cfg['vocab_size']}.txt"
    if not vocab_file.exists():
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(seq.vocab))
    with open(md / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    figs = tr.results_dir() / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    shutil.copy(tr.runs_dir() / rid / "curvas.png", figs / f"curvas_{cell}.png")

    result = {"cell": cell, "run_id": rid, "config": cfg, "params": run["params"],
              "epochs_run": run["epochs_run"], "best_epoch": run["best_epoch"],
              "train_time_sec": run["train_time_sec"], "sec_per_epoch": run["sec_per_epoch"],
              "val": run["val"], "test": test_metrics, "por_clase": per_class,
              "empty_docs": run.get("empty_docs")}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n*** {tr.CELL_NAMES[cell]} FINAL (test, evaluado una vez): exactitud={test_metrics['accuracy']:.4f} | "
          f"F1 macro={test_metrics['f1_macro']:.4f}")
    return result


def extra_seeds(cell: str, cfg: dict, seq: tr.SequenceData, class_names: list[str], seeds: list[int]) -> None:
    """La configuración final con otras semillas, para reportar media ± desviación (como la E1)."""
    with open(tr.results_dir() / "final" / f"{cell}.json", encoding="utf-8") as f:
        main_res = json.load(f)
    rows = [{"seed": cfg["seed"], **{k: main_res["test"][k] for k in ("accuracy", "f1_macro")}}]
    for s in seeds:
        res = run_or_load(dict(cfg, seed=s), seq, class_names, phase="semillas", eval_test=True)
        rows.append({"seed": s, **{k: res["test"][k] for k in ("accuracy", "f1_macro")}})
    summary = {"cell": cell, "por_semilla": rows}
    for k in ("accuracy", "f1_macro"):
        vals = np.array([r[k] for r in rows])
        summary[k] = {"media": float(vals.mean()), "desv": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    with open(tr.results_dir() / "final" / f"{cell}_semillas.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def mlp_reference(seeds: list[int], quick: bool) -> None:
    """Re-entrena el MLP Tuned de la E1 con el mismo script (corregido) y en el mismo entorno."""
    import train_mlp
    root = tr.OUTPUT_ROOT.resolve()
    with working_dir(root):
        rows = []
        for s in [42] + seeds:
            name = "tuned_mlp_e2" if s == 42 else f"tuned_mlp_e2_s{s}"
            jfile = Path("results") / f"{name}.json"
            if not jfile.exists():
                print(f"\n=== [mlp] {name} ===")
                args = (["--quick-test"] if quick else MLP_TUNED_ARGS) + ["--output-name", name, "--seed", str(s),
                                                                          "--no-demo"]
                if s == 42:
                    args += ["--probs-out", "results/e2/preds/mlp_test_probs.npy"]
                with contextlib.redirect_stdout(io.StringIO()) if quick else contextlib.nullcontext():
                    train_mlp.main(args)
                keras.backend.clear_session()
            with open(jfile, encoding="utf-8") as f:
                r = json.load(f)
            rows.append({"seed": s, "accuracy": r["accuracy"], "f1_macro": r["f1_macro"]})

        # Pesos y vectorizador para la demo (archivo de pesos: mucho más liviano que el .keras con optimizador)
        md = Path("models") / "e2"
        md.mkdir(parents=True, exist_ok=True)
        if not (md / "mlp_tuned.weights.h5").exists():
            mlp = keras.models.load_model("models/tuned_mlp_e2.keras")
            cfg = {"vocab_size": int(mlp.input_shape[-1]), "hidden_dim": int(mlp.layers[0].units),
                   "dropout": float(mlp.layers[1].rate)}
            with contextlib.redirect_stdout(io.StringIO()):
                fresh = train_mlp.build_mlp_model(cfg["vocab_size"], mlp.output_shape[-1], cfg["hidden_dim"],
                                                  cfg["dropout"])
            fresh.set_weights(mlp.get_weights())          # solo pesos de la red, sin estado del optimizador
            fresh.save_weights(md / "mlp_tuned.weights.h5")
            shutil.copy("models/vectorizer_tuned_mlp_e2.pkl", md / "mlp_vectorizer.pkl")
            with open(md / "mlp_config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        summary = {"por_semilla": rows}
        for k in ("accuracy", "f1_macro"):
            vals = np.array([r[k] for r in rows])
            summary[k] = {"media": float(vals.mean()), "desv": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        with open(Path("results") / "e2" / "final" / "mlp_semillas.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Entrega 2 completa: EDA, ajuste, modelos finales y referencia MLP")
    p.add_argument("--quick-test", action="store_true", help="Prueba de humo (pocos minutos) en ./prueba_rapida")
    p.add_argument("--output-root", default=None)
    p.add_argument("--skip-seeds", action="store_true", help="No entrena las semillas extra")
    p.add_argument("--skip-mlp", action="store_true", help="No re-entrena el MLP de la E1")
    a = p.parse_args(argv)

    quick = a.quick_test
    root = Path(a.output_root or ("prueba_rapida" if quick else "."))
    root.mkdir(parents=True, exist_ok=True)
    tr.set_output_root(root)
    tr.results_dir().mkdir(parents=True, exist_ok=True)
    (tr.results_dir() / "final").mkdir(parents=True, exist_ok=True)

    base, search, seeds = dict(BASE), SEARCH, EXTRA_SEEDS
    if quick:
        base.update(epochs=2, vocab_size=5000, seq_len=50, units=16)
        search = [("seq_len", [50, 100]), ("units", [16, 32]), ("dropout", [0.3]), ("lr", [1e-3])]
        seeds = [1]

    log_file = open(tr.results_dir() / "log_ejecucion.txt", "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    t_start = time.perf_counter()
    print(f"\n######## ENTREGA 2 - inicio ({'prueba rápida' if quick else 'ejecución completa'}) ########")
    print(json.dumps(tr.environment_info(), indent=2, ensure_ascii=False))

    data = load_20newsgroups()
    class_names = data["class_names"]
    print(f"Datos: train {len(data['train'][0])} | val {len(data['val'][0])} | test {len(data['test'][0])}")

    # 1. EDA
    eda_sequences.main(["--output-root", str(root)], data=data)

    # 2-4. Base, ajuste y modelos finales
    seq_cache: dict[int, tr.SequenceData] = {}

    def seq_for(cfg):
        V = cfg["vocab_size"]
        if V not in seq_cache:
            seq_cache[V] = tr.SequenceData(data, V)
        return seq_cache[V]

    seq = seq_for(base)
    finals = {}
    for cell in CELLS:
        print(f"\n######## {tr.CELL_NAMES[cell]}: base y ajuste de parámetros ########")
        finals[cell] = tune(cell, seq, class_names, base, search)
    for cell in CELLS:
        finalize(cell, finals[cell], seq, class_names)

    # 5. Semillas extra
    if not a.skip_seeds:
        for cell in CELLS:
            print(f"\n######## {tr.CELL_NAMES[cell]}: semillas {seeds} ########")
            extra_seeds(cell, finals[cell], seq, class_names, seeds)

    # 6. MLP de referencia
    if not a.skip_mlp:
        print("\n######## MLP Tuned (E1) en el mismo entorno ########")
        mlp_reference([] if a.skip_seeds else seeds, quick)

    minutes = (time.perf_counter() - t_start) / 60
    print(f"\n######## ENTREGA 2 - fin ({minutes:.1f} min) ########")
    for cell in CELLS:
        with open(tr.results_dir() / "final" / f"{cell}.json", encoding="utf-8") as f:
            r = json.load(f)
        print(f"{tr.CELL_NAMES[cell]:>9}: L={r['config']['seq_len']} u={r['config']['units']} "
              f"d={r['config']['dropout']} lr={r['config']['lr']:g} -> test exactitud {r['test']['accuracy']:.4f}, "
              f"F1 macro {r['test']['f1_macro']:.4f}")
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
