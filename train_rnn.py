"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 2: RNN SIN MEMORIA (SimpleRNN) Y RNN CON MEMORIA (LSTM)
DATASET 3: 20 NEWSGROUPS (20 CATEGORÍAS) | FRAMEWORK: KERAS 3 + TENSORFLOW
================================================================================
Entrena UN modelo recurrente y guarda todo lo necesario para el reporte:

  results/e2/runs/<run_id>/config.json   configuración, versiones y equipo
  results/e2/runs/<run_id>/history.json  métricas por época (para las curvas)
  results/e2/runs/<run_id>/metrics.json  métricas de validación (y de test con --eval-test)
  results/e2/runs/<run_id>/curvas.png    curvas de pérdida y exactitud (épocas desde 1)
  checkpoints/<run_id>.weights.h5        pesos de la mejor época (menor val_loss)
  results/e2/bitacora.csv                una fila por corrida

Ejemplos:
  python train_rnn.py --cell simple --quick-test
  python train_rnn.py --cell lstm --seq-len 200 --units 128 --dropout 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Al cargar pesos en un modelo nuevo, Keras avisa que no restaura el estado del optimizador (no se necesita)
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import keras
from sklearn import metrics as skm

from data_utils import load_20newsgroups


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN Y REPRODUCIBILIDAD
# ==============================================================================

# Configuración base (ver PLAN_ENTREGA2.md §4 y §5: de dónde sale cada valor)
DEFAULTS = {
    "cell": "lstm",        # "simple" (RNN sin memoria) o "lstm" (RNN con memoria)
    "seq_len": 200,        # L: tokens por documento (EDA + validación)
    "vocab_size": 20_000,  # V: incluye el relleno ('') y la palabra desconocida ([UNK])
    "embed_dim": 100,      # tamaño del embedding (ejemplo oficial de Keras con 20 Newsgroups)
    "units": 128,          # unidades de la capa recurrente (se ajusta en validación)
    "dropout": 0.3,        # dropout después de la RNN (E1 usó 0.3-0.4; se ajusta en validación)
    "lr": 1e-3,            # Adam, igual que la E1
    "clipnorm": 1.0,       # recorte del gradiente (Pascanu et al., 2013)
    "epochs": 30,          # máximo; EarlyStopping decide cuándo parar
    "batch_size": 64,      # igual que la E1
    "seed": 42,            # igual que la E1
}

CELL_NAMES = {"simple": "SimpleRNN", "lstm": "LSTM"}
COLORS = {"simple": "#2a78d6", "lstm": "#eb6834", "mlp": "#1baf7a"}  # paleta validada (dataviz)
EMPTY_TOKEN = "emptydoc"   # marcador para documentos vacíos (PLAN §3.3)

OUTPUT_ROOT = Path(".")


def set_output_root(root: str | Path) -> None:
    """Permite escribir todo bajo otra carpeta (por ejemplo, para la prueba rápida)."""
    global OUTPUT_ROOT
    OUTPUT_ROOT = Path(root)


def results_dir() -> Path:
    return OUTPUT_ROOT / "results" / "e2"


def runs_dir() -> Path:
    return results_dir() / "runs"


def checkpoints_dir() -> Path:
    return OUTPUT_ROOT / "checkpoints"


def models_dir() -> Path:
    return OUTPUT_ROOT / "models" / "e2"


def set_seed(seed: int) -> None:
    """Fija semillas para que la corrida sea reproducible (mismo criterio que la E1)."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def run_id(cfg: dict) -> str:
    """Nombre único y legible de una corrida a partir de su configuración."""
    return (f"{cfg['cell']}_L{cfg['seq_len']}_V{cfg['vocab_size']}_u{cfg['units']}"
            f"_d{cfg['dropout']}_lr{cfg['lr']:g}_s{cfg['seed']}")


def environment_info() -> dict:
    """Versiones y equipo: van al reporte (condiciones de entrenamiento)."""
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    ram_gb = None
    try:
        import psutil  # disponible en Colab
        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    gpus = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            gpus.append(tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name))
        except Exception:
            gpus.append(gpu.name)
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "sistema": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "nucleos_cpu": os.cpu_count(),
        "ram_gb": ram_gb,
        "gpu": gpus or None,
    }


# ==============================================================================
# SECCIÓN 2: DE TEXTO A SECUENCIAS
# ==============================================================================

def fill_empty(texts: list[str]) -> list[str]:
    """Documentos vacíos -> token marcador, para que ninguna secuencia quede vacía (PLAN §3.3)."""
    return [t if t.strip() else EMPTY_TOKEN for t in texts]


def to_tensor(texts: list[str]) -> tf.Tensor:
    """Cadenas -> tensor de TensorFlow (evita np.array de texto, que reserva memoria por el texto más largo)."""
    return tf.constant(texts, dtype=tf.string)


class SequenceData:
    """
    Vectorizador ajustado SOLO con train (igual que el TF-IDF de la E1) y las secuencias de
    cada partición. Se guarda la versión de longitud variable y se recorta/rellena a L cuando
    se pide, así varias corridas con distinto L reutilizan el mismo vocabulario.
    """

    def __init__(self, data: dict, vocab_size: int):
        self.vocab_size = vocab_size
        self.labels = {s: data[s][1] for s in ("train", "val", "test")}
        raw = {s: data[s][0] for s in ("train", "val", "test")}
        texts = {s: fill_empty(raw[s]) for s in raw}

        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=vocab_size, output_mode="int", ragged=True)  # minúsculas y sin puntuación (por defecto)
        self.vectorizer.adapt(to_tensor(texts["train"]))
        self.vocab = self.vectorizer.get_vocabulary()
        self.n_tokens = len(self.vocab)   # = vocab_size salvo que train tenga menos palabras distintas
        self.empty_id = self.vocab.index(EMPTY_TOKEN) if EMPTY_TOKEN in self.vocab else 1

        self.ragged = {s: self.vectorizer(to_tensor(texts[s])) for s in texts}
        lengths = {s: self.ragged[s].row_lengths().numpy() for s in texts}
        # Vacíos = texto en blanco + textos que quedan sin tokens al quitar la puntuación
        self.n_empty = {s: int(sum(not t.strip() for t in raw[s]) + int((lengths[s] == 0).sum())) for s in raw}

    def padded(self, split: str, seq_len: int) -> np.ndarray:
        """Matriz (N, L): ceros a la derecha y, si el texto es más largo, los primeros L tokens."""
        X = self.ragged[split].to_tensor(default_value=0, shape=[None, seq_len]).numpy().astype("int32")
        X[(X != 0).sum(axis=1) == 0, 0] = self.empty_id
        return X


# ==============================================================================
# SECCIÓN 3: ARQUITECTURA (SimpleRNN O LSTM, MISMO ESQUELETO)
# ==============================================================================

CELLS = {"simple": keras.layers.SimpleRNN, "lstm": keras.layers.LSTM}


def build_rnn(cell: str, vocab_size: int, emb_dim: int = 100, units: int = 128, dropout: float = 0.3,
              lr: float = 1e-3, clipnorm: float = 1.0, num_classes: int = 20) -> keras.Model:
    """
    Embedding (con máscara para ignorar el relleno) -> capa recurrente -> Dropout -> Softmax.
    Lo único que cambia entre los dos modelos es la capa recurrente: así la diferencia
    entre ellos se debe a la memoria.
    """
    inp = keras.Input(shape=(None,), dtype="int32", name="ids")
    x = keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True, name="embedding")(inp)
    x = CELLS[cell](units, name=cell)(x)              # último estado oculto
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="softmax")(x)
    model = keras.Model(inp, out, name=f"{cell}_rnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def count_params(model: keras.Model) -> dict:
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "entrenables": trainable, "no_entrenables": total - trainable}


# ==============================================================================
# SECCIÓN 4: ENTRENAMIENTO
# ==============================================================================

class EpochLogger(keras.callbacks.Callback):
    """Guarda la duración de cada época y la tasa de aprendizaje usada."""

    def on_train_begin(self, logs=None):
        self.times, self.lrs = [], []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t0)
        self.lrs.append(float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)))


def evaluate(model: keras.Model, X: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    """Mismas métricas que la E1, calculadas a partir de las probabilidades."""
    probs = model.predict(X, batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)
    loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-7, 1.0))))
    return {
        "loss": loss,
        "accuracy": float(skm.accuracy_score(y, y_pred)),
        "f1_macro": float(skm.f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(skm.f1_score(y, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(skm.precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(skm.recall_score(y, y_pred, average="macro", zero_division=0)),
    }, probs


def plot_curves(history: dict, cell: str, best_epoch: int, path: Path) -> None:
    """Curvas de pérdida y exactitud. Color = modelo; entrenamiento punteado, validación continua."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    color = COLORS[cell]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [("loss", "val_loss", "Pérdida (entropía cruzada)"), ("accuracy", "val_accuracy", "Exactitud")]
    for ax, (k_tr, k_va, label) in zip(axes, panels):
        ax.plot(epochs, history[k_tr], color=color, lw=2, ls="--", alpha=0.6, label="Entrenamiento")
        ax.plot(epochs, history[k_va], color=color, lw=2, label="Validación")
        ax.axvline(best_epoch, color="#898781", lw=1, ls=":")
        ax.annotate(f"mejor época: {best_epoch}", xy=(best_epoch, 0.98), xycoords=("data", "axes fraction"),
                    xytext=(4, -2), textcoords="offset points", fontsize=8, color="#52514e", va="top")
        ax.set_xlabel("Época")
        ax.set_ylabel(label)
        ax.set_xticks(epochs if len(epochs) <= 15 else np.arange(1, len(epochs) + 1, 2))
        ax.grid(True, color="#e1e0d9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.legend(frameon=False)
    fig.suptitle(f"{CELL_NAMES[cell]}: curvas de pérdida y exactitud", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def append_log(row: dict) -> None:
    """Una fila por corrida en la bitácora (results/e2/bitacora.csv)."""
    path = results_dir() / "bitacora.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def train_and_evaluate(cfg: dict, seq: SequenceData, class_names: list[str], phase: str = "manual",
                       eval_test: bool = False, verbose: int = 2) -> dict:
    """Entrena una configuración, carga los pesos de la mejor época y guarda todo."""
    keras.backend.clear_session()
    set_seed(cfg["seed"])
    rid = run_id(cfg)
    rdir = runs_dir() / rid
    rdir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir().mkdir(parents=True, exist_ok=True)
    ckpt = checkpoints_dir() / f"{rid}.weights.h5"

    print(f"\n=== [{phase}] {rid} ===")
    X_tr, X_va = seq.padded("train", cfg["seq_len"]), seq.padded("val", cfg["seq_len"])
    y_tr, y_va = seq.labels["train"], seq.labels["val"]

    model = build_rnn(cfg["cell"], seq.n_tokens, cfg["embed_dim"], cfg["units"], cfg["dropout"],
                      cfg["lr"], cfg["clipnorm"], num_classes=len(class_names))
    logger = EpochLogger()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True, save_weights_only=True),
        logger,
    ]
    t0 = time.perf_counter()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=cfg["epochs"],
                     batch_size=cfg["batch_size"], callbacks=callbacks, verbose=verbose)
    train_time = time.perf_counter() - t0
    model.load_weights(ckpt)   # pesos de la época con menor val_loss

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    history["lr"] = logger.lrs
    history["epoch_time_sec"] = logger.times
    best_epoch = int(np.argmin(history["val_loss"]) + 1)

    val_metrics, _ = evaluate(model, X_va, y_va)
    result = {
        "run_id": rid,
        "phase": phase,
        "config": cfg,
        "params": count_params(model),
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "sec_per_epoch": train_time / len(history["loss"]),
        "val": val_metrics,
        "empty_docs": seq.n_empty,
    }
    if eval_test:
        X_te = seq.padded("test", cfg["seq_len"])
        test_metrics, probs = evaluate(model, X_te, seq.labels["test"])
        result["test"] = test_metrics
        np.save(rdir / "test_probs.npy", probs.astype("float32"))

    with open(rdir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "phase": phase, "entorno": environment_info()}, f, indent=2)
    with open(rdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(rdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    plot_curves(history, cfg["cell"], best_epoch, rdir / "curvas.png")

    test = result.get("test", {})
    append_log({
        "run_id": rid, "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "fase": phase,
        "celda": CELL_NAMES[cfg["cell"]], "L": cfg["seq_len"], "V": cfg["vocab_size"], "unidades": cfg["units"],
        "dropout": cfg["dropout"], "lr": cfg["lr"], "semilla": cfg["seed"], "epocas": result["epochs_run"],
        "mejor_epoca": best_epoch, "val_loss": round(val_metrics["loss"], 4),
        "val_acc": round(val_metrics["accuracy"], 4), "val_f1_macro": round(val_metrics["f1_macro"], 4),
        "test_acc": round(test["accuracy"], 4) if test else "", "test_f1_macro": round(test["f1_macro"], 4) if test else "",
        "s_por_epoca": round(result["sec_per_epoch"], 2), "parametros": result["params"]["total"],
    })
    print(f"-> val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
          f"val_F1={val_metrics['f1_macro']:.4f} | mejor época={best_epoch}/{result['epochs_run']} | "
          f"{result['sec_per_epoch']:.1f} s/época" + (f" | test_acc={test['accuracy']:.4f}" if test else ""))
    return result


# ==============================================================================
# SECCIÓN 5: FUNCIÓN PRINCIPAL (CLI)
# ==============================================================================

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="RNN (SimpleRNN o LSTM) para 20 Newsgroups - Entrega 2")
    p.add_argument("--cell", choices=["simple", "lstm"], default=DEFAULTS["cell"])
    p.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    p.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    p.add_argument("--embed-dim", type=int, default=DEFAULTS["embed_dim"])
    p.add_argument("--units", type=int, default=DEFAULTS["units"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--clipnorm", type=float, default=DEFAULTS["clipnorm"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--eval-test", action="store_true", help="Evalúa en test (solo para el modelo final)")
    p.add_argument("--output-root", default=".", help="Carpeta donde se escriben results/, models/ y checkpoints/")
    p.add_argument("--quick-test", action="store_true", help="Prueba rápida: 2 épocas, V=5000, L=50")
    a = p.parse_args(argv)

    set_output_root(a.output_root)
    cfg = {"cell": a.cell, "seq_len": a.seq_len, "vocab_size": a.vocab_size, "embed_dim": a.embed_dim,
           "units": a.units, "dropout": a.dropout, "lr": a.lr, "clipnorm": a.clipnorm,
           "epochs": a.epochs, "batch_size": a.batch_size, "seed": a.seed}
    if a.quick_test:
        cfg.update(epochs=2, vocab_size=5000, seq_len=50)

    results_dir().mkdir(parents=True, exist_ok=True)
    data = load_20newsgroups()
    seq = SequenceData(data, cfg["vocab_size"])
    train_and_evaluate(cfg, seq, data["class_names"], phase="manual", eval_test=a.eval_test)


if __name__ == "__main__":
    main()
