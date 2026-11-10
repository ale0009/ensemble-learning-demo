# ADR-0003: Exportación dual de resultados en consola y archivos de evidencia

- **Estado:** Aceptado
- **Fecha:** 2026-11-09
- **Proyecto:** Comparación de clasificadores y ensambles

## Contexto

La salida debe servir tanto para inspección inmediata en terminal como para documentación automatizada.

## Decisión

Formatear tablas Markdown con tabulate para stdout y persistir resultados/comparacion.csv y resultados/comparacion.png.

## Consecuencias

Permite actualización transparente de tablas en el README y conserva artefactos gráficos reproducibles.
