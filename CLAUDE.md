# Hybrid Visual Search Model — Tesis de Licenciatura en Ciencias de Datos

## Contexto del proyecto

Modelo computacional de búsqueda visual basado en inferencia bayesiana. El modelo simula cómo un observador humano busca un objeto target en una imagen, generando scanpaths (secuencias de fijaciones oculares). El objetivo de la tesis es usar las **features extraídas de estos scanpaths como predictores de señales de EEG**.

El modelo principal se llama **ELM** (Expected Likelihood Model). Compite con una baseline bayesiana llamada **nnIBS**.

---

## Estructura del proyecto

```
.
├── run_models.py                  # Entry point principal
├── hyperparameter_search.py       # Búsqueda de hiperparámetros (experimentos del paper)
├── constants.py                   # Rutas globales y métricas disponibles
├── utils.py                       # Utilidades de I/O de resultados
├── Model/
│   ├── main.py                    # Orquesta la ejecución del modelo
│   ├── configs/
│   │   ├── elm_final.json         # Config principal del ELM (usar este para experimentos)
│   │   ├── elm_base.json          # Config base sin optimizaciones
│   │   └── nnIBS.json             # Config del modelo bayesiano clásico
│   └── visualsearch/
│       ├── visual_searcher.py     # Clase principal: orquesta el loop de búsqueda
│       ├── grid.py                # Discretización de imagen en celdas
│       ├── visibility_map.py      # Mapa de visibilidad foveal/periférica (gaussiana)
│       ├── visual_evidence_history.py  # Memoria de trabajo (WM): NoHistory, Simple, Degraded
│       ├── target_selector.py     # Selección del objeto a buscar: MinEntropy, Random, etc.
│       ├── target_absent.py       # Estrategia para target ausente (actualmente no-op)
│       ├── target_present_condition.py  # Condición de fin de trial: Oracle, ConsecutiveFix
│       ├── models/
│       │   ├── elm_model.py       # Expected Information Gain: saccade hacia max EIG
│       │   ├── bayesian_model.py  # Modelo bayesiano clásico (costoso computacionalmente)
│       │   └── greedy_model.py    # Greedy: saccade hacia el máximo del posterior
│       ├── target_similarity/
│       │   ├── target_similarity.py    # Clase base: mu, sigma, at_fixation()
│       │   ├── ivsnresnext.py     # ResNeXt101: método principal del ELM
│       │   ├── ivsn.py            # VGG16: método del nnIBS
│       │   ├── ssim.py            # SSIM: alternativa clásica
│       │   ├── correlation.py     # Cross-correlación normalizada
│       │   └── geisler.py         # Geisler et al. (2005): sin cálculo adicional
│       ├── prior/
│       │   └── prior.py           # Prior: deepgaze (saliency), uniform, noisy
│       └── utils/
│           ├── utils.py           # I/O de imágenes, scanpaths, checkpoints, CSV heatmaps
│           └── deepgaze/          # DeepGaze II para generar saliency maps (TF1)
├── Metrics/                       # Módulo de métricas: perf, mm, hsp, sa, rf
├── Datasets/                      # Datasets: People, COCOSearch18, Interiors, Unrestricted
│   └── <dataset>/
│       ├── dataset_info.json
│       ├── trials_properties.json
│       └── scanpaths/             # Scanpaths humanos por sujeto
└── Results/                       # Scanpaths generados por el modelo
```

---

## Cómo ejecutar

```bash
# Correr ELM en un dataset con todas las métricas
python run_models.py --d People --m elm_final --mts perf mm hsp

# Correr múltiples modelos y datasets
python run_models.py --d People COCOSearch18 --m elm_base elm_final nnIBS --mts perf mm

# Búsqueda de hiperparámetros (experimentos del paper)
python hyperparameter_search.py

# Correr solo un modelo directamente (para debugging)
python -m Model.main -dataset People --cfg elm_final
```

---

## Conceptos clave del modelo

### Loop de búsqueda (`visual_searcher.py`)
1. Se carga la imagen, el prior (saliency map de DeepGaze) y el memory set (targets)
2. Para cada fijación:
   - Se computa el **posterior no-normalizado** para cada objeto del memory set
   - El **target selector** elige qué objeto buscar (MinEntropy por defecto)
   - El **search model** (ELM/Bayesian/Greedy) elige la próxima fijación
   - Se verifica si el target fue encontrado (condición Oracle)
3. Se guarda el scanpath en `Results/`

### ELM Model (`elm_model.py`)
- Selecciona la próxima fijación maximizando el **Expected Information Gain (EIG)**
- `EIG(fixation) = 0.5 * Σ posterior(loc) * fovea_map(loc | fixation)`
- Mucho más rápido que el modelo bayesiano completo

### Visual Evidence History (`visual_evidence_history.py`)
- **NoHistoryVisualEvidence**: acumula toda la evidencia (sin límite de memoria)
- **SimpleHistoryVisualEvidence**: ventana deslizante de tamaño `history_size`
- **DegradedHistoryVisualEvidence**: ponderación exponencial de fixaciones pasadas ← **config del ELM final**

### Target Similarity (`target_similarity/target_similarity.py`)
- Genera mapas `mu` y `sigma` para cada celda de la grilla
- `mu[loc, fixation]`: cuánto se parece la posición `loc` al target cuando se fija en `fixation`
- `at_fixation()`: devuelve la evidencia visual en la fijación actual (con ruido gaussiano)
- El **fovea_filter** combina evidencia foveal y periférica

### Memoria de trabajo multi-objeto
- El modelo puede buscar múltiples objetos simultáneamente (memory set)
- `target_selector` decide cuál objeto del memory set buscar en cada fijación
- `searched_object_indexes`: registra qué objeto se buscó en cada fijación

---

## Configuración del ELM final (`elm_final.json`)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `search_model` | `elm` | Expected Information Gain |
| `target_similarity` | `ivsnresnext` | ResNeXt101 para similitud |
| `prior` | `deepgaze` | Saliency map neuronal |
| `cell_size` | `32` | Píxeles por celda de la grilla |
| `history_size` | `8` | Ventana de memoria de trabajo |
| `history_degradation` | `true` | Ponderación exponencial |
| `prior_as_fixation` | `true` | Prior como fijación inicial |
| `fovea_filter` | `true` | Filtro foveal/periférico activo |
| `peripheral_exponent` | `0.2` | Decaimiento periférico suave |
| `fovea_exponent` | `2` | Decaimiento foveal pronunciado |
| `target_index_selector` | `MinEntropy` | Selección por mínima entropía |

---

## Convenciones importantes

- **Coordenadas**: `row = Y`, `column = X`
- **Grid**: cada celda tiene `cell_size=32` píxeles. La imagen estándar es `768x1024 px` → grilla de `24x32` celdas
- **Sigma (visibilidad)**: `[[4000, 0], [0, 2600]]` — elipse orientada horizontal (campo visual humano)
- **Scanpaths**: guardados como JSON en `Results/<dataset>_dataset/<config>/Scanpaths.json`
- **Checkpoints**: al presionar Ctrl+C se guarda un checkpoint para reanudar
- **`filters_mss`**: lista de tamaños de memory set a **excluir** de la ejecución

---

## Outputs del modelo

Cada entrada en `Scanpaths.json` contiene:
```json
{
  "image_name.jpg": {
    "X": [col1, col2, ...],         // Coordenadas X de las fijaciones (en celdas)
    "Y": [row1, row2, ...],         // Coordenadas Y de las fijaciones (en celdas)
    "target_found": true,
    "target_bbox": [r1, c1, r2, c2],
    "elm_info_gained": [0.12, ...], // EIG por fijación ← feature clave para EEG
    "searched_object_indexes": [...],// Qué objeto se buscó en cada fijación
    "memory_set": ["obj1.jpg", ...],
    "target_object": "obj1.jpg",
    "max_fixations": 8
  }
}
```

### Features relevantes para predicción de EEG
- `elm_info_gained`: información ganada por fijación (proxy de sorpresa/P300)
- `searched_object_indexes`: cambios de objeto buscado (proxy de carga cognitiva)
- Longitud del scanpath hasta encontrar el target
- Distancia entre fijaciones consecutivas
- Entropía del posterior en cada fijación

---

## Métricas disponibles

| Código | Descripción |
|--------|-------------|
| `perf` | Cumulative performance (tasa de target encontrado por fijación) |
| `mm` | MultiMatch (comparación de scanpaths con humanos) |
| `hsp` | Human Scanpath Prediction (log-likelihood) — muy lento |
| `perfs` | Performance por tamaño de memory set |
| `mms` | MultiMatch por tamaño de memory set |
| `sa` | Sequence alignment |
| `rf` | Rank frequency |

---

## Dependencias principales

```
tensorflow==2.12.0   # DeepGaze II (saliency maps)
torch==2.0.1         # ResNeXt101 / VGG16 (target similarity)
torchvision==0.15.2
numpy==1.26.4
scipy==1.14.1
scikit-image==0.21.0
pandas==2.2.3
multimatch_gaze==0.1.3
EntropyHub               # Entropía 2D para MinEntropy2D selector
```

---

## Notas para la tesis (EEG)

- La señal de EEG se alinea temporalmente con los **onset de cada fijación**
- El `elm_info_gained` es el predictor principal: alta información ganada → mayor respuesta P300
- Los cambios en `searched_object_indexes` pueden correlacionar con componentes de atención selectiva
- Para extraer features por fijación: iterar sobre `X`, `Y`, `elm_info_gained` del scanpath
- Pipeline sugerido: `Scanpaths.json` → feature extraction → alineación con EEG epochs → modelo predictivo (PyMC / sklearn)

# Contexto de Tesis: Integración de Modelos Bayesianos de Búsqueda Visual y EEG

## 1. Visión General del Proyecto
Este proyecto busca unir dos áreas de la neurociencia computacional: la modelización de movimientos oculares y el análisis de señales cerebrales (EEG). El objetivo central es extraer características (features) del modelo de búsqueda visual **nnELM** (neural network Entropy Limit Minimization) y utilizarlas como predictores en modelos de deconvulución de EEG para predecir la actividad cerebral durante tareas de búsqueda híbrida.

## 2. Marco Teórico y Modelos
- **Búsqueda Híbrida (Hybrid Search):** Tarea donde los sujetos deben buscar uno o más objetivos (targets) de un conjunto memorizado (Memory Set Size, MSS = 1, 2, 4) dentro de escenas naturales con múltiples distractores.
- **nnELM (Modelo de Movimientos Oculares):** Un modelo bayesiano que predice la secuencia de fijaciones (scanpaths). Utiliza mapas de saliencia (DeepGaze II) como *prior* y mapas de similitud de objetivos (ResNet/ResNeXt) para la verosimilitud (*likelihood*). Se basa en la minimización de la entropía para decidir la siguiente fijación.
- **Análisis de EEG (Deconvolución):** Se utilizan modelos lineales regularizados (Ridge Regression) para estimar Funciones de Respuesta Temporal (TRFs). El objetivo es desentrelazar las respuestas neuronales solapadas asociadas a fijaciones y sacadas consecutivas.

## 3. Características (Features) a Extraer (nnELM -> EEG)
Se han identificado features clave para operacionalizar procesos cognitivos en el modelo de EEG:
- **Bottom-Up:** Valor del mapa de saliencia en las coordenadas de la fijación.
- **Target Similarity:** Similitud visual entre la fijación actual y el target memorizado.
- **Evidencia Visual (W):** Certidumbre del sistema en la fijación actual.
- **Probabilidad Posterior:** Creencia bayesiana actualizada sobre la ubicación del target tras T fijaciones.
- **Métricas de Entropía:** Entropía del prior, entropía global de la posterior y ganancia de información (Divergencia KL entre T-1 y T).
- **Competencia:** Ratios de entropía y densidad de candidatos para cuantificar conflictos en la selección de la próxima sacada.

## 4. Metodología de Integración
1. **Identificación:** Extraer las features dinámicas del modelo nnELM fijación por fijación.
2. **Alineación (Mapping):** Asignar los valores calculados por celdas en el modelo a las fijaciones reales registradas en el experimento de EEG (40 sujetos). 
   - *Desafío:* Manejar casos donde múltiples fijaciones caen en una misma celda del modelo.
3. **Encoding:** Correr modelos de encoding de deconvulución utilizando las features del modelo como predictores adicionales a las variables basales (amplitud de sacada, rango de fijación, etc.).

## 5. Objetivos Técnicos en Claude
- Validar la lógica de transferencia de features entre dominios.
- Asistir en la implementación de scripts en Python (MNE, PyMC, Scikit-learn).
- Interpretar resultados de performance ($R^2$, AIC) y colinealidad (VIF).
- Estructurar secciones de la tesis con rigor académico.

## 6. Referencias Principales
- **Ruarte et al. (2025):** "Integrating Bayesian and neural networks models for eye movement prediction in hybrid search".
- **Care et al. (MS):** "Time-space signatures of hybrid search resolution using EEG and eye movements concurrent recordings".
- **Bujia et al. (2022):** "Modeling Human Visual Search in Natural Scenes".