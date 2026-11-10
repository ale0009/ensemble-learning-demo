# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y este proyecto se adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-01

### Añadido
- Implementación del componente `modelos`: Instanciación y parametrización de clasificadores individuales (KNN, Regresión Logística, Árbol, Naive Bayes) y ensambles (Voting, Bagging, Random Forest, Gradient Boosting, Stacking)..
- Implementación del componente `evaluacion`: Ejecución de validación cruzada estratificada en cinco particiones y cálculo de media y desviación estándar para accuracy y F1 macro..
- Implementación del componente `cli`: Orquestación del benchmarking, renderizado de tablas en consola en formato Markdown y exportación de comparacion.csv y comparacion.png..
- Implementación del componente `__main__`: Punto de entrada de línea de comandos para despachar las instrucciones del módulo CLI..
- Configuración de flujo de trabajo de integración continua.
- Documentación técnica, guías de arquitectura y políticas de seguridad.
