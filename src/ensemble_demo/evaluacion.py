"""Evaluación estratificada, reproducible y presentación de métricas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate

from ensemble_demo.modelos import SEMILLA, crear_modelos


def cargar_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Devuelve dos datasets integrados y uno sintético con semilla fija."""
    cancer = load_breast_cancer()
    wine = load_wine()
    sintetico = make_classification(
        n_samples=900, n_features=20, n_informative=12, n_redundant=4,
        n_classes=3, n_clusters_per_class=2, class_sep=1.0,
        random_state=SEMILLA,
    )
    return {
        "Breast cancer": (cancer.data, cancer.target),
        "Wine": (wine.data, wine.target),
        "Sintético": sintetico,
    }


def evaluar_modelos(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    modelos: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Calcula media y desviación estándar de accuracy y F1 macro en 5 folds."""
    datos = cargar_datasets() if datasets is None else datasets
    candidatos = crear_modelos() if modelos is None else modelos
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas: list[dict[str, float | str]] = []

    for dataset, (x, y) in datos.items():
        for nombre, modelo in candidatos.items():
            scores = cross_validate(
                modelo, x, y, cv=cv, scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                n_jobs=1, error_score="raise",
            )
            filas.append({
                "dataset": dataset,
                "modelo": nombre,
                "accuracy_media": float(np.mean(scores["test_accuracy"])),
                "accuracy_desviacion": float(np.std(scores["test_accuracy"], ddof=1)),
                "f1_macro_media": float(np.mean(scores["test_f1_macro"])),
                "f1_macro_desviacion": float(np.std(scores["test_f1_macro"], ddof=1)),
            })
    return pd.DataFrame(filas)


def tabla_markdown(resultados: pd.DataFrame) -> str:
    """Devuelve resultados ordenados por F1 macro medio, de mayor a menor."""
    columnas = ["dataset", "modelo", "accuracy_media", "accuracy_desviacion", "f1_macro_media", "f1_macro_desviacion"]
    tabla = resultados.loc[:, columnas].sort_values(
        ["dataset", "f1_macro_media"], ascending=[True, False]
    ).copy()
    for columna in columnas[2:]:
        tabla[columna] = tabla[columna].map(lambda valor: f"{valor:.4f}")
    tabla = tabla.rename(columns={
        "dataset": "Dataset", "modelo": "Modelo", "accuracy_media": "Accuracy media",
        "accuracy_desviacion": "Desv. accuracy", "f1_macro_media": "F1 macro medio",
        "f1_macro_desviacion": "Desv. F1 macro",
    })
    return tabla.to_markdown(index=False)
