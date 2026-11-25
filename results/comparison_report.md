# Comparación de Desempeño: Experimentos MLP (Dataset 3: 20 Newsgroups)

Este reporte técnico compara cuantitativamente el desempeño de diferentes configuraciones de red neuronal MLP.

| Métrica / Parámetro | baseline_mlp | tuned_mlp |
| :--- | :---: | :---: |
| **Tamaño Vocabulario TF-IDF** | 5000 | 10000 |
| **Neuronas (Capa 1)** | 256 | 512 |
| **Tasa Dropout** | 0.4 | 0.3 |
| **Regularización L2** | 0.0001 | 5e-05 |
| **Learning Rate** | 0.001 | 0.0005 |
| **Batch Size** | 64 | 64 |
| --- | --- | --- |
| **Exactitud (Test Accuracy)** | 0.6446 | 0.6803 |
| **F1-Score (Macro)** | 0.6296 | 0.6690 |
| **F1-Score (Weighted)** | 0.6424 | 0.6806 |
| **Precisión (Macro)** | 0.6481 | 0.6830 |
| **Sensibilidad (Macro Recall)** | 0.6303 | 0.6679 |
| **Tiempo de Entrenamiento** | 35.69s | 133.42s |
| **Épocas Máximas** | 20 | 25 |
