"""
map_loader.py — Carga de mapas estáticos y dinámicos del modelo nnELM.

Todos los paths son relativos al directorio Notebooks/ (donde corren los notebooks).

Mapas estáticos (independientes de la fijación):
    saliencia  : prior bottom-up (DeepGaze II) — un mapa por imagen
    similitud  : evidencia visual top-down (ResNeXt101) — un mapa por (imagen, target)

Mapas dinámicos (extraídos en cada fijación humana, archivo .npz por sujeto/imagen):
    visual_evidence  (n_fix, 24, 32)  evidencia W acumulada hasta cada fijación
    posterior        (n_fix, 24, 32)  creencia bayesiana posterior tras cada fijación
    entropy_map      (n_fix, 24, 32)  -posterior * log(posterior) — entropía por celda
    expected_ig_map  (n_fix, 24, 32)  EIG esperado si la próxima fijación fuera cada celda (Eq. 6 del paper)
    fixations_y      (n_fix,)         fila en la grilla (coord Y, 0-23)
    fixations_x      (n_fix,)         columna en la grilla (coord X, 0-31)
    memory_set       list[str]        objetos del memory set del trial
    target_stim      str              target buscado
    target_found     bool
"""

from pathlib import Path
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
PATHS = {
    'human_maps': Path('../Human_Maps/HSUBANOTT'),
    'saliencia':  Path('../Results/saliency_maps/HSUBANOTT/deepgaze'),
    'similitud':  Path('../Results/target_similarity_maps/HSUBANOTT/Ivsnresnext'),
    'originales': Path('../Datasets/HSEM/images'),
    'targets':    Path('../Datasets/HSEM/stimuli'),
}

# Dimensiones de la grilla del modelo (imagen 768×1024 px, cell_size=32)
GRID_SHAPE = (24, 32)   # (filas, columnas)


# ===========================================================================
# MAPAS ESTÁTICOS
# ===========================================================================

def cargar_imagen_original(image_name: str) -> np.ndarray:
    """Imagen original RGB. Shape: (H, W, 3)."""
    return np.array(Image.open(PATHS['originales'] / image_name).convert('RGB'))


def cargar_imagen_target(target_name: str) -> np.ndarray:
    """Imagen del target en RGB. Shape: (H, W, 3)."""
    return np.array(Image.open(PATHS['targets'] / target_name).convert('RGB'))


def cargar_mapa_saliencia(image_name: str, como_grilla: bool = False) -> np.ndarray:
    """
    Mapa de saliencia bottom-up (DeepGaze II), normalizado a [0, 1].

    Args:
        image_name : nombre del archivo de imagen (ej: 'cmp_building_014_person_007.jpg')
        como_grilla: si True, redimensiona a GRID_SHAPE (24, 32) para alinear
                     con los mapas dinámicos en el espacio de la grilla del modelo

    Returns:
        np.ndarray float32: shape (H, W) si como_grilla=False, (24, 32) si True
    """
    img = Image.open(PATHS['saliencia'] / image_name).convert('L')
    if como_grilla:
        img = img.resize((GRID_SHAPE[1], GRID_SHAPE[0]), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def cargar_mapa_similitud(image_name: str, target_name: str,
                           como_grilla: bool = False) -> np.ndarray:
    """
    Mapa de similitud visual top-down (ResNeXt101), normalizado a [0, 1].

    El archivo en disco se nombra: {image_stem}_{target_stem}.png
    (ej: 'cmp_building_014_person_007_person385.png')

    Args:
        image_name : nombre de la imagen  (ej: 'cmp_building_014_person_007.jpg')
        target_name: nombre del target    (ej: 'person385.png')
        como_grilla: si True, redimensiona a GRID_SHAPE (24, 32)

    Returns:
        np.ndarray float32: shape (H, W) o (24, 32)

    Raises:
        FileNotFoundError: si el mapa no existe para este par (imagen, target)
    """
    fname = f'{Path(image_name).stem}_{Path(target_name).stem}.png'
    img = Image.open(PATHS['similitud'] / fname).convert('L')
    if como_grilla:
        img = img.resize((GRID_SHAPE[1], GRID_SHAPE[0]), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def mapa_similitud_existe(image_name: str, target_name: str) -> bool:
    """Verifica si el mapa de similitud existe para este par (imagen, target)."""
    fname = f'{Path(image_name).stem}_{Path(target_name).stem}.png'
    return (PATHS['similitud'] / fname).exists()


# ===========================================================================
# MAPAS DINÁMICOS
# ===========================================================================

def cargar_mapas_humanos(subject_id: str, image_name: str) -> dict:
    """
    Carga los mapas internos del modelo extraídos en las fijaciones humanas.

    Args:
        subject_id : ID del sujeto (ej: 'S101', 'et_117969')
        image_name : nombre de la imagen (ej: 'cmp_building_014_person_007.jpg')

    Returns:
        dict con claves:
            visual_evidence : (n_fix, 24, 32) float32
            posterior       : (n_fix, 24, 32) float32
            entropy_map     : (n_fix, 24, 32) float32  o  None (archivos viejos)
            expected_ig_map : (n_fix, 24, 32) float32  o  None (archivos anteriores a esta versión)
            fixations_y     : (n_fix,) int16  — fila en la grilla
            fixations_x     : (n_fix,) int16  — columna en la grilla
            memory_set      : list[str]
            target_stim     : str
            target_found    : bool

    Nota:
        entropy_map.sum(axis=(1, 2)) da la entropía escalar por fijación.
        expected_ig_map[i, y, x] es el EIG que obtendría el modelo si fijara en (y, x)
        tras la i-ésima fijación. El máximo de cada mapa indica la sácada óptima del modelo.
        Usar entropia_escalar(data) y eig_en_fijacion(data) para derivar features escalares.
    """
    npz_path = PATHS['human_maps'] / subject_id / (Path(image_name).stem + '.npz')
    raw = np.load(npz_path, allow_pickle=True)

    data = {k: raw[k] for k in raw.files}

    # Compatibilidad: archivos anteriores tenían 'entropy' escalar, no 'entropy_map'
    if 'entropy' in data and 'entropy_map' not in data:
        data['entropy_scalar'] = data.pop('entropy')
        data['entropy_map'] = None

    # Compatibilidad: archivos anteriores a la versión con expected_ig_map
    if 'expected_ig_map' not in data:
        data['expected_ig_map'] = None

    data['memory_set']   = list(data['memory_set'])
    data['target_stim']  = str(data['target_stim'])
    data['target_found'] = bool(data['target_found'])
    return data


# ===========================================================================
# FEATURES DERIVADAS
# ===========================================================================

def entropia_escalar(data: dict) -> np.ndarray:
    """
    Entropía de Shannon del posterior en cada fijación. Shape: (n_fix,).

    Equivale a entropy_map.sum(axis=(1, 2)).
    Compatible con archivos nuevos (entropy_map 2D) y viejos (entropy escalar).
    """
    if data.get('entropy_map') is not None:
        return data['entropy_map'].sum(axis=(1, 2))
    return data.get('entropy_scalar', np.array([]))


def kl_divergencia_consecutiva(data: dict) -> np.ndarray:
    """
    Divergencia KL entre posteriors consecutivos: KL(P_t || P_{t-1}).

    Mide cuánta información aportó cada fijación al actualizar la creencia.
    Análogo al elm_info_gained del modelo, pero calculado sobre los posteriors reales.

    Shape: (n_fix - 1,)  — la fijación 0 no tiene referencia anterior.
    """
    post = data['posterior']   # (n_fix, 24, 32)
    eps = 1e-12
    return np.sum(post[1:] * np.log((post[1:] + eps) / (post[:-1] + eps)), axis=(1, 2))


def eig_en_fijacion(data: dict) -> np.ndarray:
    """
    EIG asignado por el modelo a la celda donde el humano realmente fijó. Shape: (n_fix,).

    Mide cuán "informativamente óptima" fue cada fijación humana según el modelo:
    valores altos indican que el humano fijó donde el modelo hubiera predicho mayor
    ganancia de información.

    Requiere archivos nuevos (con expected_ig_map). Retorna array vacío si no disponible.
    """
    eig_map = data.get('expected_ig_map')
    if eig_map is None:
        return np.array([])
    return valor_en_fijacion(eig_map, data)


def eig_maximo(data: dict) -> np.ndarray:
    """
    Máximo del expected_ig_map en cada fijación — EIG de la sácada óptima. Shape: (n_fix,).

    Refleja cuánta información podría haber ganado el modelo en el mejor movimiento posible.
    Útil como referencia de la "oportunidad" de aprendizaje en cada momento del trial.
    """
    eig_map = data.get('expected_ig_map')
    if eig_map is None:
        return np.array([])
    return eig_map.max(axis=(1, 2))


def concentracion_posterior(data: dict) -> np.ndarray:
    """
    Máximo del posterior en cada fijación — proxy de certeza/concentración.

    Shape: (n_fix,).  Aumenta conforme el modelo se acerca al target.
    """
    return data['posterior'].max(axis=(1, 2))


def valor_en_fijacion(mapa: np.ndarray, data: dict) -> np.ndarray:
    """
    Extrae el valor de un mapa en la celda de cada fijación.

    Args:
        mapa : (n_fix, 24, 32) o (24, 32)
               Si es 2D, aplica la misma grilla a todas las fijaciones.
        data : dict devuelto por cargar_mapas_humanos

    Returns:
        np.ndarray (n_fix,) — valor del mapa en cada fijación
    """
    ys = data['fixations_y']
    xs = data['fixations_x']
    if mapa.ndim == 2:
        return mapa[ys, xs]
    return mapa[np.arange(len(ys)), ys, xs]


def tabla_features_fijacion(data: dict, image_name: str) -> 'pd.DataFrame':
    """
    Construye un DataFrame con una fila por fijación y las features candidatas para EEG.

    Requiere que el archivo sea de versión nueva (con entropy_map).

    Columnas:
        fixation        índice de fijación (0-based)
        fix_y, fix_x    coordenadas en la grilla
        entropy         entropía escalar del posterior
        kl_prev         KL vs. posterior anterior (NaN en fijación 0)
        max_posterior   concentración (máximo del posterior)
        ve_at_fix       evidencia visual en la celda fijada
        post_at_fix     probabilidad posterior en la celda fijada
        eig_at_fix      EIG del modelo en la celda fijada (NaN si archivo viejo)
        eig_max         EIG máximo del mapa — sácada óptima del modelo (NaN si archivo viejo)
    """
    import pandas as pd

    n = data['posterior'].shape[0]
    kl  = kl_divergencia_consecutiva(data)
    kl_col = np.concatenate([[np.nan], kl])

    eig_fix = eig_en_fijacion(data)
    eig_fix_col = eig_fix if len(eig_fix) == n else np.full(n, np.nan)
    eig_max_col = eig_maximo(data)
    eig_max_col = eig_max_col if len(eig_max_col) == n else np.full(n, np.nan)

    return pd.DataFrame({
        'fixation':      np.arange(n),
        'fix_y':         data['fixations_y'],
        'fix_x':         data['fixations_x'],
        'entropy':       entropia_escalar(data),
        'kl_prev':       kl_col,
        'max_posterior': concentracion_posterior(data),
        've_at_fix':     valor_en_fijacion(data['visual_evidence'], data),
        'post_at_fix':   valor_en_fijacion(data['posterior'], data),
        'eig_at_fix':    eig_fix_col,
        'eig_max':       eig_max_col,
    })


# ===========================================================================
# HELPERS DE EXPLORACIÓN
# ===========================================================================

def listar_sujetos() -> list:
    """Lista ordenada de sujetos con mapas extraídos."""
    return sorted([p.name for p in PATHS['human_maps'].iterdir() if p.is_dir()])


def listar_imagenes(subject_id: str) -> list:
    """Lista ordenada de imágenes disponibles para un sujeto (extensión .jpg)."""
    return sorted([p.stem + '.jpg' for p in (PATHS['human_maps'] / subject_id).glob('*.npz')])


def resumen_disponibilidad() -> 'pd.DataFrame':
    """
    DataFrame con la cantidad de imágenes y fijaciones por sujeto.

    Columnas: subject_id, n_imagenes, n_fijaciones_total, n_fijaciones_media
    """
    import pandas as pd

    filas = []
    for subj in listar_sujetos():
        imagenes = listar_imagenes(subj)
        total_fix = 0
        for img in imagenes:
            try:
                d = cargar_mapas_humanos(subj, img)
                total_fix += d['posterior'].shape[0]
            except Exception:
                pass
        filas.append({
            'subject_id':         subj,
            'n_imagenes':         len(imagenes),
            'n_fijaciones_total': total_fix,
            'n_fijaciones_media': round(total_fix / len(imagenes), 1) if imagenes else 0,
        })
    return pd.DataFrame(filas)
