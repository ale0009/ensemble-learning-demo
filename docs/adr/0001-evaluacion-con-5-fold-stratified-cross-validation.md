# ADR-0001: Evaluación con 5-fold Stratified Cross-Validation

- **Estado:** Aceptado
- **Fecha:** 2026-11-09
- **Proyecto:** Comparación de clasificadores y ensambles

## Contexto

Se requiere comparar modelos y ensambles de manera balanceada en conjuntos de datos pequeños.

## Decisión

Implementar validación cruzada estratificada de 5 particiones computando media y desviación estándar de accuracy y F1 macro.

## Consecuencias

Conserva la proporción original de clases en cada fold y proporciona una estimación confiable de variabilidad.
