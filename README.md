# Proyecto M5GP para Clasificación

## Descripción del Proyecto

Durante este proyecto, se desarrollaron diversos módulos para dotar al M5GP de la capacidad de realizar tareas de clasificación. Se añadieron cuatro modelos base que pueden ser seleccionados para su uso en combinación con el M5GP:

- **Regresión Logística**
- **K Vecinos Cercanos**
- **Máquinas de Soporte Vectorial** (tanto en memoria como en lote)
- **Bosque Aleatorio**

Además, se incorporaron diversas métricas de evaluación que pueden seleccionarse según la tarea específica a realizar. Esto es particularmente útil para conjuntos de datos desbalanceados.

Finalmente, se evaluó el desempeño del M5GP en combinación con regresión logística y bosque aleatorio, demostrando su efectividad y potencial en diferentes contextos de clasificación.

## Características

- **Módulos de Clasificación**: Implementación de cuatro modelos base para combinar con M5GP.
- **Métricas de Evaluación**: Diversas métricas seleccionables según las necesidades del conjunto de datos.
- **Evaluación de Desempeño**: Análisis detallado del rendimiento de M5GP combinado con regresión logística y bosque aleatorio.

## Requisitos del Sistema

- Python 3.x
- Bibliotecas necesarias (especificadas en `requirements.txt`)

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/tu-usuario/proyecto-m5gp.git
    ```
2. Navega al directorio del proyecto:
    ```bash
    cd proyecto-m5gp
    ```
3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```