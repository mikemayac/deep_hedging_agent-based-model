# Modelo Chiarella-Heston para Deep Hedging

## Descripción General

Este es el proyecto final para la materia de Modelación Basada en Angentes que se centra en calibrar el **modelo Chiarella-Heston** utilizando datos históricos del índice Nikkei 225 y utilizando el modelo calibrado para generar datos de entrenamiento para estrategias de **deep hedging**. El rendimiento de estas estrategias se compara con el delta hedging tradicional y el deep hedging basado en el modelo de Movimiento Browniano Geométrico (GBM).

El proyecto busca demostrar cómo los modelos basados en agentes, específicamente el modelo Chiarella-Heston, pueden mejorar la efectividad de las estrategias de deep hedging al capturar hechos estilizados de los mercados financieros que los modelos tradicionales como GBM no logran reproducir.


## Estructura del Proyecto

```
├── Chiarella_Heston_Deeper_Hedging.ipynb  # Notebook principal con análisis y experimentos
├── calibration.py                         # Script para calibración del modelo
├── helpers.py                             # Funciones auxiliares para simulaciones y cálculos
├── records/
│   ├── abm0/                              # Resultados y modelos entrenados con datos GBM
│   ├── abm1/                              # Resultados y modelos entrenados con Chiarella-Heston
├── README.md                              # Documentación del proyecto
└── requirements.txt                       # Lista de dependencias
```

## Instalación

### Prerequisitos

- Python 3.7 o superior
- Git (para clonar el repositorio)


### Crear un Entorno Virtual (Opcional pero Recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows usar `venv\Scripts\activate`
```

### Instalar Dependencias

```bash
pip install -r requirements.txt
```

## Uso

El proyecto está estructurado alrededor de un Jupyter Notebook para exploración interactiva y scripts de Python para la calibración del modelo.

### 1. Preparación de Datos

Asegúrate de tener acceso a los datos históricos del índice Nikkei 225. El notebook usa la biblioteca `yfinance` para descargar estos datos automáticamente.

### 2. Calibración del Modelo

El script `calibration.py` se usa para calibrar el modelo Chiarella-Heston.

#### Ejecutar Calibración

```bash
python calibration.py
```

- **Nota**: La calibración puede ser computacionalmente intensiva y puede llevar un tiempo significativo de unas 35hrs.

- Los resultados de la calibración se guardan en el directorio `records/`.

### 3. Experimentos de Deep Hedging

Los experimentos principales se realizan en el notebook `chiarella_heston_deep_hedging.ipynb`.

#### Abrir el Notebook

```bash
jupyter notebook chiarella_heston_deep_hedging.ipynb
```

#### Secciones del Notebook

1. **Datos**: Carga y analiza datos históricos del índice Nikkei 225.

2. **Resultados de Calibración**: Utiliza parámetros pre-calibrados para simular trayectorias de precios.

3. **Comparación de Modelos**: Compara los modelos GBM, Chiarella Extendido y Chiarella-Heston en la reproducción de hechos estilizados.

4. **Experimentos de Deep Hedging**: Entrena y evalúa agentes de deep hedging usando los datos simulados.

#### Notas Importantes

- El notebook usa resultados de calibración pre-generados (archivos `.pkl` en `records/abm0` y `records/abm1`) para ahorrar tiempo.

- Si deseas recalibrar los modelos o modificar parámetros, asegúrate de ajustar las secciones correspondientes en `helpers.py` y volver a ejecutar `calibration.py`.

## Resultados

- **Rendimiento del Modelo**: El modelo Chiarella-Heston captura mejor los hechos estilizados del mercado en comparación con los modelos GBM y Chiarella Extendido.

- **Estrategias de Deep Hedging**: Los agentes entrenados con datos del modelo Chiarella-Heston superan a los entrenados con datos GBM, especialmente bajo costos de transacción más altos.

- **Gestión de Riesgos**: Los agentes de deep hedging basados en el modelo Chiarella-Heston muestran mejor manejo del riesgo, evidenciado por un menor Déficit Esperado en condiciones adversas del mercado.

## Dependencias

El proyecto requiere las siguientes bibliotecas de Python:

- numpy
- pandas
- matplotlib
- seaborn
- yfinance
- statsmodels
- scipy
- pickle
- PyTorch (para modelos de deep learning)
- jupyter

Instala todas las dependencias usando:

```bash
pip install -r requirements.txt
```

