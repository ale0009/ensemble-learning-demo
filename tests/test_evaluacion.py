"""Pruebas de reproducibilidad y contrato de evaluación."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ensemble_demo.evaluacion import cargar_datasets, evaluar_modelos
from ensemble_demo.modelos import crear_modelos


def test_evaluacion_es_reproducible() -> None:
    """La semilla fija debe producir exactamente las mismas métricas."""
    x, y = make_classification(n_samples=120, n_features=8, random_state=7)
    datos = {"prueba": (x, y)}
    modelos = {"logistica": LogisticRegression(max_iter=500)}
    primero = evaluar_modelos(datos, modelos)
    segundo = evaluar_modelos(datos, modelos)
    pd.testing.assert_frame_equal(primero, segundo)


def test_modelos_entrenan_en_datasets_incluidos() -> None:
    """Cada modelo debe ajustarse sin error a cada dataset del proyecto."""
    modelos = crear_modelos()
    for x, y in cargar_datasets().values():
        for modelo in modelos.values():
            modelo.fit(x, y)
            assert len(modelo.predict(x)) == len(y)


def test_tabla_tiene_columnas_esperadas() -> None:
    """La salida tabular conserva los nombres acordados para CSV y CLI."""
    x, y = make_classification(n_samples=100, n_features=6, random_state=5)
    resultado = evaluar_modelos({"prueba": (x, y)}, {"logistica": LogisticRegression()})
    assert list(resultado.columns) == [
        "dataset", "modelo", "accuracy_media", "accuracy_desviacion",
        "f1_macro_media", "f1_macro_desviacion",
    ]
    assert np.isfinite(resultado.select_dtypes(include="number").to_numpy()).all()
