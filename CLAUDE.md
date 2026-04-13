# Tesis: Hybrid Visual Search (ELM) & EEG Integration

## 1. Core Concept
Modelado de búsqueda visual bayesiana (**ELM** vs **nnIBS**) para predecir señales de EEG.
- **Input**: Imágenes naturales + Memory Set (MSS 1, 2, 4).
- **Output Model**: Scanpaths, EIG (Expected Info Gain), Posteriores.
- **Goal**: Usar métricas del modelo como predictores (Features) para TRFs de EEG (Ridge Regression/Deconvolución).

## 2. Technical Stack & Structure
- **Stack**: Python (TF 2.12, Torch 2.0, MNE, Sklearn).
- **Grilla**: 24x32 celdas (cell_size=32px sobre 768x1024).
- **Files Clave**:
  - `visual_searcher.py`: Loop principal.
  - `elm_model.py`: Cálculo de EIG (proxy de sorpresa cognitiva).
  - `visual_evidence_history.py`: WM (DegradedHistory = config final).
  - `target_similarity/`: Extracción de features (ResNeXt101).

## 3. Key EEG Predictors (Features)
| Feature | Model Source | Neuro-Hypothesis |
|---------|--------------|------------------|
| **Saliency** | DeepGaze II | Bottom-up attention |
| **Similarity** | ResNeXt | Target template matching |
| **EIG / KL Div** | ELM Model | Information Gain / P300 |
| **Entropy** | Global Posterior | Uncertainty / Cognitive Load |
| **Switch** | target_selector | Object switching cost |

## 4. Integration Workflow (Mapping)
1. Extraer features por fijación desde `Results/Scanpaths.json`.
2. Mapear celdas del modelo (Row, Col) a fijaciones reales de sujetos (ET).
3. **Encoding**: Ridge Regression para desentrelazar solapamiento de señales en EEG.

## 5. Implementation Notes
- **Coords**: `(Y, X)` o `(Row, Col)`.
- **Config Final**: `elm_final.json` (History size 8, fovea_filter ON).
- **Metrics**: `perf`, `mm` (MultiMatch), `hsp`.

---

## ⚠️ Interaction Rules (Token Efficiency)
1. **Be Concise**: Responde en bullet points. No repitas el contexto a menos que haya cambios.
2. **Code**: Solo fragmentos relevantes. Evita boilerplate de imports a menos que sean nuevos.
3. **No Lectures**: No expliques qué es PCA o Bayes si no te lo pido; asume nivel Data Science.
4. **Direct Answer**: Si pregunto por un error, ve directo a la causa en el código.