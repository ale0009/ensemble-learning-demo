# ADR-0002: Uso exclusivo de datasets integrados y generados sintéticamente

- **Estado:** Aceptado
- **Fecha:** 2026-11-09
- **Proyecto:** Comparación de clasificadores y ensambles

## Contexto

El experimento debe ser ejecutable sin conectividad de red ni descarga de archivos externos.

## Decisión

Emplear los datasets wine y breast_cancer de scikit-learn junto con make_classification.

## Consecuencias

Garantiza una suite de ejecución totalmente autónoma, rápida y sin riesgo de cambios en recursos remotos.
