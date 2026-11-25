"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 2: DEMO EJECUTABLE (MLP de la E1 vs. SimpleRNN vs. LSTM)
================================================================================
Clasifica el mismo texto con los tres modelos y muestra el top-3 de cada uno.
No necesita internet: usa los archivos de models/e2/ que deja run_e2.py.

Modos:
  python demo_e2.py --run-samples        casos de la E1 + par de orden + texto largo
  python demo_e2.py --text "..."         un texto
  python demo_e2.py                      modo interactivo
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import tensorflow as tf

import train_rnn as tr

MODELS = Path("models") / "e2"

SAMPLES = [
    ("NASA rover detects evidence of liquid water and atmosphere on Mars.", "Caso claro (E1)"),
    ("The engine needs a new oil filter and spark plugs for better mileage.", "Caso claro (E1)"),
    ("A debate on whether doctors should perform experimental medical treatments when religious beliefs oppose it.",
     "Caso mixto (E1): medicina vs. religión"),
    ("I installed a new operating system on my PC to play baseball and hockey video games.",
     "Caso mixto (E1): sistemas vs. deportes"),
    ("Congressional debate on government funding for public healthcare hospitals and health insurance.",
     "Caso mixto (E1): política vs. salud"),
    ("The doctor treated the patient without medicine.", "Par de orden (1/2)"),
    ("The patient treated the doctor without medicine.", "Par de orden (2/2)"),
]
FILLER = "I have been thinking about this for a while and I wanted to share a few thoughts with everyone here. "
KEY_SENTENCE = "Anyway, the goalie made forty saves and the hockey team won the playoff game in overtime."


class Classifier:
    def __init__(self, name: str, predict_fn, seq_len: int | None = None):
        self.name, self.predict_fn, self.seq_len = name, predict_fn, seq_len


def load_rnn(cell: str, class_names: list[str]) -> Classifier | None:
    cfg_file = MODELS / f"{cell}_final_config.json"
    if not cfg_file.exists():
        return None
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    vocab = (MODELS / f"vocab_V{cfg['vocab_size']}.txt").read_text(encoding="utf-8").split("\n")
    vec = keras.layers.TextVectorization(output_mode="int", vocabulary=vocab, output_sequence_length=cfg["seq_len"])
    model = tr.build_rnn(cell, len(vocab), cfg["embed_dim"], cfg["units"], cfg["dropout"],
                         num_classes=len(class_names))
    model.load_weights(MODELS / f"{cell}_final.weights.h5")
    empty_id = vocab.index(tr.EMPTY_TOKEN) if tr.EMPTY_TOKEN in vocab else 1

    def predict(texts):
        # Mismo preprocesamiento que en el entrenamiento (train_rnn.SequenceData.padded)
        X = vec(tf.constant(tr.fill_empty(texts))).numpy()
        X[(X != 0).sum(axis=1) == 0, 0] = empty_id
        return model.predict(X, verbose=0)

    return Classifier(tr.CELL_NAMES[cell], predict, cfg["seq_len"])


def load_mlp(class_names: list[str]) -> Classifier | None:
    cfg_file = MODELS / "mlp_config.json"
    if not cfg_file.exists():
        return None
    import train_mlp
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        model = train_mlp.build_mlp_model(cfg["vocab_size"], len(class_names), cfg["hidden_dim"], cfg["dropout"])
    model.load_weights(MODELS / "mlp_tuned.weights.h5")
    with open(MODELS / "mlp_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    def predict(texts):
        return model.predict(vectorizer.transform(texts).toarray(), verbose=0)

    return Classifier("MLP (E1)", predict)


def show(text: str, label: str, classifiers: list[Classifier], class_names: list[str], n_tokens: int | None = None):
    print("\n" + "=" * 100)
    print(f"[{label}]")
    print(f"\"{text if len(text) < 160 else text[:110] + ' [...] ' + text[-60:]}\"")
    if n_tokens is not None:
        print(f"Palabras del texto: {n_tokens}")
    print("-" * 100)
    print(f"{'Modelo':<12}{'1.ª opción':<32}{'2.ª opción':<32}{'3.ª opción':<32}")
    for c in classifiers:
        probs = c.predict_fn([text])[0]
        top = np.argsort(probs)[::-1][:3]
        cells = [f"{class_names[i]} ({probs[i] * 100:.1f} %)" for i in top]
        note = f"   (solo lee las primeras {c.seq_len})" if (c.seq_len and n_tokens and n_tokens > c.seq_len) else ""
        print(f"{c.name:<12}" + "".join(f"{s:<32}" for s in cells) + note)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Demo Entrega 2: MLP vs. SimpleRNN vs. LSTM (20 Newsgroups)")
    p.add_argument("--text", default=None, help="Texto en inglés a clasificar")
    p.add_argument("--run-samples", action="store_true", help="Casos preparados para la sustentación")
    a = p.parse_args(argv)

    class_names = json.loads((MODELS / "class_names.json").read_text(encoding="utf-8"))
    classifiers = [c for c in (load_mlp(class_names), load_rnn("simple", class_names), load_rnn("lstm", class_names))
                   if c is not None]
    print("Modelos cargados: " + ", ".join(c.name for c in classifiers))

    if a.run_samples:
        for text, label in SAMPLES:
            show(text, label, classifiers, class_names)
        max_L = max((c.seq_len or 0) for c in classifiers)
        long_text = FILLER * max(1, (max_L // 20) + 2) + KEY_SENTENCE
        show(long_text, "Texto largo: las palabras clave aparecen al final", classifiers, class_names,
             n_tokens=len(long_text.split()))
        return
    if a.text:
        show(a.text, "Texto", classifiers, class_names, n_tokens=len(a.text.split()))
        return

    print("\nEscribe un texto en inglés y presiona ENTER ('salir' para terminar).")
    while True:
        try:
            text = input("\nTexto > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue
        if text.lower() in {"salir", "exit", "quit"}:
            break
        show(text, "Texto", classifiers, class_names, n_tokens=len(text.split()))


if __name__ == "__main__":
    main()
