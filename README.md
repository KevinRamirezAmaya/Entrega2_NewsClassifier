# Clasificación de Texto en 20 Newsgroups con Keras: MLP (Entrega 1) y RNN (Entrega 2)

Proyecto Práctico - Redes Neuronales (Universidad del Valle, 2026-2)
John Ramos · Kevin Ramirez · Dylan Morales

* **Entrega 1 (15 %):** Perceptrón Multicapa (MLP) sobre vectores TF-IDF.
* **Entrega 2 (20 %):** RNN **sin memoria** (`SimpleRNN`) y RNN **con memoria** (`LSTM`) sobre la secuencia de palabras.

Las dos entregas usan exactamente los mismos datos: 20 Newsgroups (versión de scikit-learn) sin encabezados,
firmas ni citas; partición oficial por fecha (11 314 train / 7 532 test) y 15 % del train para validación
(estratificado, semilla 42) → 9 616 / 1 698 / 7 532 documentos.

---

## 1. Qué hay en esta carpeta

```text
NewsClassifier/
├── README.md
├── requirements.txt
├── data_utils.py          # Carga y partición compartidas por la E1 y la E2
├── train_mlp.py           # E1: MLP con TF-IDF (con las correcciones de la E2, ver §6)
├── compare_results.py     # E1: compara Baseline vs. Tuned
├── demo_inference.py      # E1: demo del MLP
├── eda_sequences.py       # E2: longitudes, vocabulario y documentos vacíos (justifica L y V)
├── train_rnn.py           # E2: entrena una SimpleRNN o una LSTM con la configuración dada
├── run_e2.py              # E2: ejecución completa (EDA, ajuste, modelos finales, semillas, MLP de referencia)
├── analyze_e2.py          # E2: tablas (LaTeX/Markdown/CSV) y figuras del informe
├── memoria_e2.py          # E2: prueba de memoria con distancia controlada (usa los modelos guardados)
├── demo_e2.py             # E2: demo que compara MLP, SimpleRNN y LSTM sobre el mismo texto
├── notebooks/
│   ├── Entrega2_Colab.ipynb        # Corre toda la E2 en Google Colab (GPU) y descarga los resultados
│   └── Demo_Entrega2_Colab.ipynb   # Demo en Colab, sin instalar nada
├── results/               # E1: métricas y curvas del MLP · E2: todo en results/e2/
├── models/e2/             # Pesos, configuración y vocabulario de los modelos finales (los usa la demo)
└── informes/
    ├── Entrega1/          # RN1.pdf (informe de la E1) + RN1.tex y sus figuras
    └── Entrega2/          # RN2.pdf (informe de la E2) + RN2.tex, figuras y tablas
```

Cada carpeta de `informes/` compila sola (`compilar.bat` con Tectonic, `pdflatex` u Overleaf).

## 2. Requisitos

* Python 3.10 o superior.
* `pip install -r requirements.txt`

En Windows, TensorFlow ≥ 2.11 no usa la GPU; la CPU alcanza para la demo y la prueba de memoria. Para
entrenar, lo más rápido es Google Colab con GPU: `notebooks/Entrega2_Colab.ipynb`.

---

## 3. Entrega 2: cómo reproducir todo

```bash
python run_e2.py --quick-test     # prueba de humo (pocos minutos, escribe en ./prueba_rapida)
python run_e2.py                  # ejecución completa (unos 25 minutos en una GPU T4)
python analyze_e2.py              # tablas y figuras del informe
python memoria_e2.py              # prueba de memoria (segundos; solo usa los modelos guardados)
python demo_e2.py --run-samples   # demo (también: --text "..." o modo interactivo)
```

`run_e2.py` hace, en orden:

1. **EDA** (`eda_sequences.py`): longitud de los documentos en tokens, % de documentos que se truncan con cada L,
   cobertura del vocabulario y documentos vacíos.
2. **Configuración base** de SimpleRNN y LSTM con los mismos hiperparámetros: Embedding(20 000, 100, `mask_zero=True`)
   → capa recurrente de 128 unidades → Dropout 0.3 → Dense(20, softmax); Adam (lr 1e-3, `clipnorm=1.0`), lote 64,
   hasta 30 épocas con EarlyStopping (val_loss, paciencia 4) y ReduceLROnPlateau (×0.5, paciencia 2).
3. **Ajuste de parámetros uno a la vez, con la semilla 42 fija**: L ∈ {100, 200, 400}, unidades ∈ {64, 128, 256},
   dropout ∈ {0.2, 0.3, 0.5}, lr ∈ {1e-3, 5e-4}. Se elige por F1 macro de validación y un cambio se adopta
   solo si mejora más de 0.5 puntos.
4. **Semillas**: con la configuración final ya fijada, cada modelo se entrena también con las semillas 1, 7 y 123.
   Se conserva como modelo final la semilla con mejor F1 macro de **validación** (fue la 42 en los tres modelos)
   y se reportan todas (`results/e2/tables/por_semilla.md`). Cada modelo se evalúa una sola vez en test; el test
   no se usa para escoger nada. El informe (sección «Uso de las semillas») explica por qué se repitió con varias semillas.
5. **MLP Tuned de la E1** re-entrenado con `train_mlp.py` en el mismo entorno (referencia y demo).

Si se interrumpe, al volver a ejecutarlo salta las corridas que ya terminaron.

**Salidas principales**

| Archivo | Contenido |
| :--- | :--- |
| `results/e2/eda.json` | Estadísticas de la EDA |
| `results/e2/bitacora.csv` | Una fila por corrida (configuración, métricas de validación y test, tiempo) |
| `results/e2/ajuste_simple.json`, `ajuste_lstm.json` | Cada paso del ajuste: valores probados, elegido y por qué |
| `results/e2/final/*.json` | Métricas de test de los modelos finales y de las semillas |
| `results/e2/runs/<run_id>/` | Configuración, versiones, historial por época y curvas de cada corrida |
| `results/e2/tables/` | Tablas del informe (`.tex`, `.md`, `.csv`), entre ellas `por_semilla` |
| `results/e2/figures/` | Figuras del informe |
| `results/e2/memoria.json` | Prueba de memoria: probabilidad de la clase correcta según la distancia |
| `results/e2/log_ejecucion.txt` | Salida completa de la ejecución en Colab |
| `models/e2/` | Pesos, configuración y vocabulario de los modelos finales (la demo funciona sin internet) |

**Resultados (conjunto de prueba, 7 532 documentos; detalle en `informes/Entrega2/RN2.pdf`)**

| Modelo | Configuración final | Exactitud (%) | F1 macro (%) | F1 macro, media de 4 semillas (%) |
| :--- | :--- | ---: | ---: | ---: |
| MLP Tuned (E1, re-entrenado) | TF-IDF 10 000, 512 unidades | 67.74 | 66.58 | 66.33 ± 0.17 |
| SimpleRNN | L = 400, 256 unidades, dropout 0.3, lr 1e-3 | 25.21 | 23.97 | 20.73 ± 2.43 |
| LSTM | L = 400, 128 unidades, dropout 0.3, lr 1e-3 | 50.08 | 49.32 | 46.11 ± 2.97 |

Exactitud y F1 macro de la semilla 42, la de mejor F1 macro de validación entre las cuatro. La LSTM duplica a la
SimpleRNN, pero ninguna RNN supera al MLP con TF-IDF de la E1. Entrenado en Google Colab (GPU Tesla T4).

---

## 4. Entrega 1: MLP con TF-IDF

```bash
python train_mlp.py --quick-test --output-name test_run
python train_mlp.py --vocab-size 5000 --hidden-dim 256 --dropout 0.4 --epochs 20 --output-name baseline_mlp
python train_mlp.py --vocab-size 10000 --hidden-dim 512 --dropout 0.3 --l2-reg 0.00005 --lr 0.0005 --epochs 25 --output-name tuned_mlp
python compare_results.py --model-a results/baseline_mlp.json --model-b results/tuned_mlp.json
python demo_inference.py --model-path models/tuned_mlp.keras --vectorizer-path models/vectorizer_tuned_mlp.pkl --run-samples
```

Argumentos de `train_mlp.py`: `--vocab-size`, `--hidden-dim`, `--dropout`, `--l2-reg`, `--lr`, `--epochs`,
`--batch-size`, `--output-name`, `--quick-test`, y desde la E2 `--seed`, `--probs-out` y `--no-demo`.

## 5. Informes y presentación

* `informes/Entrega1/RN1.pdf`: informe de la Entrega 1 con las correcciones (ver §6).
* `informes/Entrega2/RN2.pdf`: informe de la Entrega 2.
* La presentación de la Entrega 2 (10 minutos) se exporta como PDF desde Claude y se guarda en `informes/Entrega2/`.


## 7. Repositorio

https://github.com/Drytec/NewsClassifier
