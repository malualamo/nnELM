import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import matplotlib.patches as patches
import json
import glob
import random

# --- CONFIGURACIÓN DE RUTAS POR DEFECTO ---
PATHS = {
    'saliencia': '../Results/saliency_maps/HSUBANOTT/deepgaze/',
    'originales': '../Datasets/HSEM/images/',
    'targets': '../Datasets/HSEM/stimuli/',
    'similitud': '../Results/target_similarity_maps/HSUBANOTT/Ivsnresnext/'
}

# --- FUNCIONES AUXILIARES (MODULARIDAD) ---

def _cargar_en_eje(ax, ruta, titulo, cmap=None):
    """Helper modular para cargar una imagen en un eje de matplotlib con manejo de errores."""
    try:
        if os.path.exists(ruta) and not ruta.endswith('None'):
            img = Image.open(ruta)
            ax.imshow(img, cmap=cmap)
            ax.set_title(titulo, fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"No encontrado:\n{os.path.basename(ruta)}", 
                    ha='center', va='center', color='red', fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "Error de carga", ha='center', va='center', color='orange')
    ax.axis('off')

def _obtener_seleccionados(path, archivos, extensiones=('.jpg', '.jpeg', '.png')):
    """Normaliza la entrada de archivos a una lista, con selección aleatoria si es None."""
    if archivos is None:
        todos = [f for f in os.listdir(path) if f.lower().endswith(extensiones)]
        if not todos:
            return []
        return random.sample(todos, min(3, len(todos)))
    return [archivos] if isinstance(archivos, str) else archivos

# --- FUNCIONES PRINCIPALES ---

def comparacion_mapa_saliencia(archivos=None):
    seleccionados = _obtener_seleccionados(PATHS['saliencia'], archivos)
    if not seleccionados: return print("Sin imágenes de saliencia.")

    fig, axes = plt.subplots(len(seleccionados), 2, figsize=(12, 4 * len(seleccionados)), squeeze=False)

    for i, nombre in enumerate(seleccionados):
        _cargar_en_eje(axes[i, 0], os.path.join(PATHS['originales'], nombre), f"Original: {nombre}")
        _cargar_en_eje(axes[i, 1], os.path.join(PATHS['saliencia'], nombre), "Saliencia (DeepGaze)", cmap='inferno')

    plt.tight_layout()
    plt.show()

def comparacion_mapa_similitud(archivos=None):
    seleccionados = _obtener_seleccionados(PATHS['similitud'], archivos)
    if not seleccionados: return

    fig, axes = plt.subplots(len(seleccionados), 3, figsize=(15, 4 * len(seleccionados)), squeeze=False)

    for i, nombre_mapa in enumerate(seleccionados):
        nombre_sin_ext = os.path.splitext(nombre_mapa)[0]
        partes = nombre_sin_ext.split('_')
        
        # Lógica de parsing extraída
        nombre_target = partes[-1] + '.png'
        nombre_img_original = "_".join(partes[:-1]) + '.jpg'
        
        config = [
            (os.path.join(PATHS['originales'], nombre_img_original), "Original", None),
            (os.path.join(PATHS['targets'], nombre_target), "Target", None),
            (os.path.join(PATHS['similitud'], nombre_mapa), "Similitud", 'viridis')
        ]

        for j, (ruta, titulo, cmap) in enumerate(config):
            _cargar_en_eje(axes[i, j], ruta, f"{titulo}:\n{os.path.basename(ruta)}", cmap)

    plt.tight_layout()
    plt.show()

def graficar_comparativa_completa(datos, 
                                 path_originales='../Datasets/HSEM/images/',
                                 path_targets='../Datasets/HSEM/stimuli/',
                                 path_saliencia='../Results/saliency_maps/HSUBANOTT/deepgaze/',
                                 path_similitud='../Results/target_similarity_maps/HSUBANOTT/Ivsnresnext/'):
    """
    Genera una grilla comparativa con nombres de archivo exactos.
    """
    ensayos = [datos] if isinstance(datos, tuple) and not isinstance(datos[0], tuple) else datos
    num_filas = len(ensayos)
    
    fig, axes = plt.subplots(num_filas, 4, figsize=(20, 5 * num_filas), squeeze=False)

    for i, (nombre_base_img, id_target) in enumerate(ensayos):
        # CONSTRUCCIÓN DEL NOMBRE EXACTO (Igual que en comparacion_mapa_similitud)
        # Se asume el formato: nombreImagen_nombreTarget.png
        nombre_exacto_similitud = f"{nombre_base_img}_{id_target}.png"
        
        config = [
            (os.path.join(path_originales, f"{nombre_base_img}.jpg"), f"Original:\n{nombre_base_img}", None),
            (os.path.join(path_targets, f"{id_target}.png"), f"Target:\n{id_target}", None),
            (os.path.join(path_saliencia, f"{nombre_base_img}.jpg"), "Saliencia (BU)\nDeepGaze", 'inferno'),
            (os.path.join(path_similitud, nombre_exacto_similitud), "Similitud (TD)\nIvsnresnext", 'viridis')
        ]

        print(f"\n--- Procesando Fila {i+1} ---")
        for j, (ruta, titulo, cmap) in enumerate(config):
            ax = axes[i, j]
            # Imprimir el path exacto que se está intentando cargar
            print(f"{titulo.split(':')[0]}: {os.path.abspath(ruta)}")
            
            try:
                if os.path.exists(ruta):
                    img = Image.open(ruta)
                    ax.imshow(img, cmap=cmap)
                    ax.set_title(titulo, fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f"No encontrado:\n{os.path.basename(ruta)}", 
                            ha='center', va='center', color='red', fontsize=8)
            except Exception:
                ax.text(0.5, 0.5, "Error de carga", ha='center', va='center', color='orange')
            
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def graficar_scanpath(scanpaths_dict, image_name, path_images='../Datasets/HSEM/images/'):
    """
    Dibuja un scanpath y resalta la ubicación del target mediante un rectángulo (bbox).
    """
    if image_name not in scanpaths_dict:
        print(f"Error: No hay datos para {image_name}")
        return

    info = scanpaths_dict[image_name]
    img_path = os.path.join(path_images, image_name)
    
    try:
        img = io.imread(img_path)
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en {img_path}")
        return
        
    height, width, _ = img.shape

    # 1. Factores de escala (Grilla 32x24 -> Pixeles reales)
    scale_y = height / info.get('image_height', 24)
    scale_x = width / info.get('image_width', 32)

    # 2. Configurar gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # 3. Dibujar el Target Bounding Box (si existe)
    if 'target_bbox' in info:
        # Formato: [y_min, x_min, y_max, x_max]
        y_min, x_min, y_max, x_max = info['target_bbox']
        
        # Convertir a coordenadas de píxeles
        rect_x = x_min * scale_x
        rect_y = y_min * scale_y
        rect_w = (x_max - x_min) * scale_x
        rect_h = (y_max - y_min) * scale_y
        
        # Crear rectángulo (estilo académico para tesis)
        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                 linewidth=3, edgecolor='red', facecolor='none', 
                                 linestyle='--', label='Target Area')
        ax.add_patch(rect)
        ax.text(rect_x, rect_y - 5, "TARGET", color='red', fontweight='bold', fontsize=10)

    # 4. Dibujar Scanpath (Fijaciones)
    xs_pixels = [x * scale_x for x in info['X']]
    ys_pixels = [y * scale_y for y in info['Y']]

    for i in range(1, len(xs_pixels)):
        ax.annotate("", xy=(xs_pixels[i], ys_pixels[i]), 
                    xytext=(xs_pixels[i-1], ys_pixels[i-1]),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2, alpha=0.6))

    for i, (x, y) in enumerate(zip(xs_pixels, ys_pixels)):
        color = 'lime' if i == 0 else ('red' if i == len(xs_pixels)-1 and info.get('target_found') else 'blue')
        circle = plt.Circle((x, y), radius=18, color=color, alpha=0.7, edgecolor='white')
        ax.add_patch(circle)
        ax.text(x, y, str(i), color='white', weight='bold', ha='center', va='center', fontsize=9)

    ax.set_title(f"Scanpath: {image_name} | Target: {'Encontrado' if info.get('target_found') else 'No encontrado'}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def cargar_datos_humanos(path_humanos, image_name):
    humanos_que_vieron_img = []
    # Buscamos todos los archivos .json en la carpeta
    archivos_json = glob.glob(os.path.join(path_humanos, "*.json"))
    
    # Debug: ¿Estamos encontrando los archivos JSON?
    if not archivos_json:
        print(f"ERROR: No se encontraron archivos JSON en la ruta: {os.path.abspath(path_humanos)}")
        return []

    target_name = image_name.strip()

    for archivo in archivos_json:
        try:
            with open(archivo, 'r') as f:
                data = json.load(f)
                # Limpiamos las llaves del JSON al comparar
                # Esto soluciona problemas de espacios o saltos de línea ocultos
                for key in data.keys():
                    if key.strip() == target_name:
                        humanos_que_vieron_img.append(data[key])
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")
    
    if not humanos_que_vieron_img:
        print(f"DEBUG: No se encontró la imagen '{target_name}' en ninguno de los {len(archivos_json)} archivos analizados.")
        # Opcional: imprimir las primeras 3 llaves del primer archivo para comparar visualmente
        if archivos_json:
             with open(archivos_json[0], 'r') as f:
                 primeras_llaves = list(json.load(f).keys())[:3]
                 print(f"DEBUG: Ejemplo de llaves encontradas en el JSON: {primeras_llaves}")

    return humanos_que_vieron_img

def _dibujar_scanpath_en_eje(ax, info, img, titulo):
    """Helper para dibujar un scanpath y el target bbox en un eje específico."""
    h, w, _ = img.shape
    scale_x = w / info.get('image_width', 1280)
    scale_y = h / info.get('image_height', 1024)

    ax.imshow(img)
    
    # 1. Dibujar Target BBox (Rojo)
    if 'target_bbox' in info:
        y_min, x_min, y_max, x_max = info['target_bbox']
        rect = patches.Rectangle((x_min * scale_x, y_min * scale_y), 
                                 (x_max - x_min) * scale_x, (y_max - y_min) * scale_y, 
                                 linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # 2. Dibujar Sacadas y Fijaciones
    xs = [x * scale_x for x in info['X']]
    ys = [y * scale_y for y in info['Y']]

    for i in range(1, len(xs)):
        ax.annotate("", xy=(xs[i], ys[i]), xytext=(xs[i-1], ys[i-1]),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5, alpha=0.6))

    for i, (x, y) in enumerate(zip(xs, ys)):
        # Inicio verde, final rojo (si encontró), resto azul
        color = 'lime' if i == 0 else ('red' if i == len(xs)-1 and info.get('target_found') else 'blue')
        ax.add_patch(plt.Circle((x, y), radius=15, color=color, alpha=0.7))
        ax.text(x, y, str(i), color='white', fontsize=7, ha='center', va='center', weight='bold')

    ax.set_title(titulo, fontsize=10)
    ax.axis('off')

def comparar_scanpaths_modelo_vs_humanos(image_name, model_scanpaths, 
                                         path_humanos='../Datasets/HSEM/human_scanpaths/', 
                                         path_images='../Datasets/HSEM/images/'):
    """
    Muestra en una fila la comparación: Modelo vs 3 Humanos aleatorios.
    """
    # 1. Obtener datos
    humanos = cargar_datos_humanos(path_humanos, image_name)
    if len(humanos) < 3:
        print(f"Aviso: Solo se encontraron {len(humanos)} humanos para esta imagen.")
    
    seleccionados_humanos = random.sample(humanos, min(3, len(humanos)))
    model_data = model_scanpaths.get(image_name)
    
    if not model_data:
        return print(f"Error: No hay datos de modelo para {image_name}")

    # 2. Cargar imagen base
    img_path = os.path.join(path_images, image_name)
    img = io.imread(img_path)

    # 3. Preparar Grid (1 fila x hasta 4 columnas)
    n_cols = 1 + len(seleccionados_humanos)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1: axes = [axes] # Manejo de caso con un solo eje

    # 4. Graficar Modelo
    _dibujar_scanpath_en_eje(axes[0], model_data, img, f"Modelo: {model_data['subject']}")

    # 5. Graficar Humanos
    for idx, h_data in enumerate(seleccionados_humanos):
        _dibujar_scanpath_en_eje(axes[idx+1], h_data, img, f"Humano: {h_data['subject']}")

    plt.tight_layout()
    plt.show()