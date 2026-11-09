"""Interfaz de línea de comandos para la comparación."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ensemble_demo.evaluacion import evaluar_modelos, tabla_markdown


def guardar_grafico(resultados: pd.DataFrame, destino: Path) -> None:
    """Guarda barras de F1 macro por dataset con desviación como error."""
    datasets = list(resultados["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(12, 5 * len(datasets)), 6), squeeze=False)
    for eje, dataset in zip(axes[0], datasets):
        grupo = resultados[resultados["dataset"] == dataset].sort_values("f1_macro_media")
        eje.barh(grupo["modelo"], grupo["f1_macro_media"], xerr=grupo["f1_macro_desviacion"], capsize=3)
        eje.set_title(dataset)
        eje.set_xlabel("F1 macro medio (± desviación estándar)")
        eje.set_xlim(0, 1.05)
        eje.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Ejecuta la suborden comparar y guarda tabla y gráfico."""
    parser = argparse.ArgumentParser(prog="python -m ensemble_demo")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    comparar = subparsers.add_parser("comparar", help="Compara todos los modelos por validación cruzada")
    comparar.add_argument("--salida", type=Path, default=Path("resultados"))
    argumentos = parser.parse_args()
    if argumentos.comando == "comparar":
        resultados = evaluar_modelos()
        argumentos.salida.mkdir(parents=True, exist_ok=True)
        resultados.to_csv(argumentos.salida / "comparacion.csv", index=False)
        guardar_grafico(resultados, argumentos.salida / "comparacion.png")
        print(tabla_markdown(resultados))
