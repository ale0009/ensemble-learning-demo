# Comparación de clasificadores y ensambles

## Problema

¿Combinar clasificadores mejora la predicción? La respuesta depende del conjunto de datos, la calidad de cada modelo y de si cometen errores distintos. Este proyecto compara modelos individuales con métodos de ensamble usando una evaluación reproducible.

## Enfoque

Se comparan árbol de decisión, regresión logística, KNN, Naive Bayes, Voting hard y soft, Bagging, Random Forest, Gradient Boosting y Stacking. Un ensamble agrega predicciones de varios modelos: Voting hard elige la clase más votada; Voting soft promedia probabilidades. Bagging entrena copias sobre muestras bootstrap; Random Forest añade selección aleatoria de variables. Gradient Boosting incorpora modelos secuencialmente para corregir errores anteriores. Stacking aprende un modelo final a partir de predicciones de otros modelos.

La validación cruzada estratificada divide los ejemplos en cinco particiones conservando la proporción de clases. Se informa la media y desviación estándar de accuracy y F1 macro. Accuracy es la fracción de aciertos; F1 macro promedia el F1 de cada clase con el mismo peso. Las medias resumen rendimiento y las desviaciones muestran variación entre particiones.

Voting es útil cuando los modelos son diversos y sus probabilidades son razonables. Bagging y Random Forest suelen reducir la variación de árboles. Gradient Boosting puede capturar patrones complejos, con mayor sensibilidad a parámetros. Stacking sirve cuando una combinación aprendida puede aprovechar fortalezas complementarias, aunque requiere validación cuidadosa.

Los datos son `breast_cancer` y `wine`, incluidos en scikit-learn, y un conjunto generado con `make_classification`; no se descargan datos.

## Cómo ejecutarlo

Requiere Python 3.11 y uv.

```bash
uv sync
uv run pytest -q
uv run python -m ensemble_demo comparar
```

El CLI imprime una tabla Markdown y escribe `resultados/comparacion.csv` y `resultados/comparacion.png`.

## Resultados

Cifras medidas con `uv run python -m ensemble_demo comparar` (cinco folds estratificados; las cifras son media ± desviación estándar). Esta tabla es la salida del CLI para cada combinación de dataset y modelo.

<!-- RESULTADOS_CLI_INICIO -->
| Dataset       | Modelo              |   Accuracy media |   Desv. accuracy |   F1 macro medio |   Desv. F1 macro |
|:--------------|:--------------------|-----------------:|-----------------:|-----------------:|-----------------:|
| Breast cancer | Regresión logística |           0.9737 |           0.0186 |           0.9714 |           0.0205 |
| Breast cancer | Stacking            |           0.9701 |           0.0133 |           0.9676 |           0.0147 |
| Breast cancer | KNN                 |           0.9631 |           0.02   |           0.9599 |           0.022  |
| Breast cancer | Voting hard         |           0.9578 |           0.0114 |           0.9547 |           0.012  |
| Breast cancer | Voting soft         |           0.9578 |           0.0144 |           0.9542 |           0.016  |
| Breast cancer | Random Forest       |           0.9543 |           0.0114 |           0.9509 |           0.0125 |
| Breast cancer | Bagging             |           0.9543 |           0.02   |           0.9509 |           0.022  |
| Breast cancer | Gradient Boosting   |           0.9491 |           0.0266 |           0.9446 |           0.0297 |
| Breast cancer | Naive Bayes         |           0.9297 |           0.0223 |           0.9244 |           0.0233 |
| Breast cancer | Árbol de decisión   |           0.928  |           0.0258 |           0.9215 |           0.0304 |
| Sintético     | Stacking            |           0.8067 |           0.0357 |           0.8061 |           0.0366 |
| Sintético     | Random Forest       |           0.8022 |           0.0187 |           0.8005 |           0.0202 |
| Sintético     | KNN                 |           0.7867 |           0.035  |           0.7868 |           0.0356 |
| Sintético     | Bagging             |           0.7633 |           0.0228 |           0.7619 |           0.0236 |
| Sintético     | Voting soft         |           0.7544 |           0.0253 |           0.7507 |           0.0291 |
| Sintético     | Gradient Boosting   |           0.7456 |           0.0339 |           0.7439 |           0.0363 |
| Sintético     | Voting hard         |           0.7111 |           0.0331 |           0.7103 |           0.0348 |
| Sintético     | Regresión logística |           0.6511 |           0.0455 |           0.645  |           0.0497 |
| Sintético     | Árbol de decisión   |           0.6378 |           0.0405 |           0.6328 |           0.0448 |
| Sintético     | Naive Bayes         |           0.6322 |           0.0392 |           0.6286 |           0.0433 |
| Wine          | Voting hard         |           0.9832 |           0.0154 |           0.9841 |           0.0146 |
| Wine          | Stacking            |           0.9832 |           0.0249 |           0.984  |           0.0241 |
| Wine          | Regresión logística |           0.9833 |           0.0152 |           0.9829 |           0.0156 |
| Wine          | Naive Bayes         |           0.9775 |           0.0309 |           0.9788 |           0.029  |
| Wine          | Random Forest       |           0.9775 |           0.0238 |           0.9784 |           0.0219 |
| Wine          | Voting soft         |           0.9717 |           0.0401 |           0.9727 |           0.0387 |
| Wine          | KNN                 |           0.9717 |           0.0202 |           0.9721 |           0.02   |
| Wine          | Bagging             |           0.9662 |           0.037  |           0.9679 |           0.0342 |
| Wine          | Gradient Boosting   |           0.9214 |           0.0367 |           0.9237 |           0.0339 |
| Wine          | Árbol de decisión   |           0.8932 |           0.0425 |           0.896  |           0.0404 |
<!-- RESULTADOS_CLI_FIN -->

## Limitaciones

Los resultados dependen de las semillas, parámetros y datasets elegidos. Tres datasets pequeños no representan todos los problemas reales. La desviación entre cinco folds no sustituye una evaluación externa, y elegir modelos mirando los mismos resultados puede sesgar la comparación.

## Origen

Basado en un trabajo académico en equipo de la Universidad ECCI, hecho con mi compañero [**David Alexander Rincón Corredor (Kuronoma)**](https://www.linkedin.com/in/davidalexanderrinconcorredorkuronoma/) y equipo; esta es una reimplementación individual y abierta.
