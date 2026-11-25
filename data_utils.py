"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
UTILIDADES DE DATOS COMPARTIDAS (ENTREGA 1 Y ENTREGA 2)
================================================================================
Carga 20 Newsgroups exactamente como en la Entrega 1 (mismos documentos, misma
limpieza y misma partición), para que el MLP y las RNN se puedan comparar.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Se quitan encabezados, firmas y citas: scikit-learn lo recomienda para que la
# evaluación sea realista (los metadatos delatan el grupo de origen).
REMOVE = ("headers", "footers", "quotes")
SPLIT_SEED = 42   # Semilla de partición de la E1 (no cambia nunca)
VAL_SIZE = 0.15   # E1: 15 % del train oficial para validación (estratificado)


def load_20newsgroups(data_home: str | None = None) -> dict:
    """
    Devuelve un diccionario con las tres particiones de la E1:
      train: 9 616 documentos | val: 1 698 | test: 7 532 (test oficial, separado por fecha)
    Cada partición es una tupla (lista_de_textos, etiquetas_int32).
    """
    raw_train = fetch_20newsgroups(subset="train", remove=REMOVE, random_state=SPLIT_SEED, data_home=data_home)
    raw_test = fetch_20newsgroups(subset="test", remove=REMOVE, random_state=SPLIT_SEED, data_home=data_home)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        raw_train.data, raw_train.target,
        test_size=VAL_SIZE, random_state=SPLIT_SEED, stratify=raw_train.target,
    )
    return {
        "train": (list(train_texts), np.asarray(train_labels, dtype=np.int32)),
        "val": (list(val_texts), np.asarray(val_labels, dtype=np.int32)),
        "test": (list(raw_test.data), np.asarray(raw_test.target, dtype=np.int32)),
        "class_names": list(raw_train.target_names),
    }
