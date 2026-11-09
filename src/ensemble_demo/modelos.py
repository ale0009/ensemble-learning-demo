"""Fábrica de modelos base y de ensamble."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 42


def crear_modelos() -> dict[str, Any]:
    """Crea modelos con parámetros y semillas fijos para la comparación."""
    arbol = DecisionTreeClassifier(max_depth=5, random_state=SEMILLA)
    logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEMILLA))
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    naive_bayes = make_pipeline(StandardScaler(), GaussianNB())
    base = [("arbol", arbol), ("logistica", logistica), ("knn", knn), ("naive_bayes", naive_bayes)]

    return {
        "Árbol de decisión": arbol,
        "Regresión logística": logistica,
        "KNN": knn,
        "Naive Bayes": naive_bayes,
        "Voting hard": VotingClassifier(estimators=base, voting="hard"),
        "Voting soft": VotingClassifier(estimators=base, voting="soft"),
        "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEMILLA), n_estimators=100, random_state=SEMILLA, n_jobs=1),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEMILLA, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEMILLA),
        "Stacking": StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1),
    }
