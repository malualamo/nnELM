"""
Extrae mapas internos del modelo nnELM en las fijaciones de los scanpaths humanos.

Para cada fijación de cada sujeto, guarda en formato .npz:
  - visual_evidence : (n_fix, grid_rows, grid_cols)  — evidencia visual W en cada fijación
  - posterior       : (n_fix, grid_rows, grid_cols)  — distribución posterior del modelo
  - entropy_map     : (n_fix, grid_rows, grid_cols)  — mapa de entropía por celda (-p·log p)
  - fixations_y     : (n_fix,)                       — fila en la grilla (coord Y)
  - fixations_x     : (n_fix,)                       — columna en la grilla (coord X)
  - memory_set      : lista de objetos en el memory set
  - target_stim     : nombre del target
  - target_found    : bool

Nota: para trials donde el target NO fue encontrado, la última fijación del humano
no tiene mapas asociados (el modelo no computa el posterior en esa fijación).

Uso:
    # Procesar todos los sujetos
    python extract_human_maps.py --d HSEM --cfg elm_final

    # Procesar un sujeto específico
    python extract_human_maps.py --d HSEM --cfg elm_final --s et_117969
"""

import argparse
import numpy as np
from os import path, makedirs, listdir

# Metrics debe inicializarse antes que visual_searcher para evitar import circular
# (mismo orden que run_models.py)
import Metrics.main  # noqa: F401

from Model.scripts import loader
from Model.scripts import constants as model_constants
from Model.visualsearch import visual_searcher as vs


MAPS_OUTPUT_DIR = 'Human_Maps'


class HumanMapExtractor(vs.VisualSearcherSubject):
    """
    Subclase de VisualSearcherSubject que sigue los scanpaths humanos
    y captura los mapas internos del modelo en cada fijación.

    Activa _capture_fixation_maps para habilitar el hook on_fixation_computed
    definido en VisualSearcher, sin modificar el flujo normal del modelo.
    """

    _capture_fixation_maps = True  # Activa el hook en VisualSearcher.search()

    def __init__(self, config, dataset_info, trials_properties, output_path,
                 human_scanpaths, sigma, filters_mss, subject_id):
        super().__init__(
            config, dataset_info, trials_properties, output_path,
            human_scanpaths, sigma, filters_mss,
            follow_human_scanpath=True
        )
        self.subject_id = subject_id
        self.maps_dir = path.join(MAPS_OUTPUT_DIR, dataset_info['dataset_name'], subject_id)
        self._maps_buffer = {}  # {image_name: {visual_evidence:[], posterior:[], ...}}

    def plot_heatmap(self):
        # Desactiva el guardado de posteriors en CSV — usamos .npz en su lugar
        return False

    def should_skip(self, image_name):
        # Saltar si el .npz ya existe (permite reanudar corridas parciales)
        out_file = path.join(self.maps_dir, image_name[:-4] + '.npz')
        if path.exists(out_file):
            print(f'  Ya existe {out_file}. Saltando.')
            return True
        return False

    def save_metrics(self, image_name):
        # No guardar métricas HSP — no son necesarias para la extracción de mapas
        pass

    def on_fixation_computed(self, fixation_number, current_fixation, visual_evidence, posterior, image_name):
        """Acumula los mapas de cada fijación en el buffer."""
        # Mapa de entropía: contribución por celda — shape (grid_rows, grid_cols)
        # La entropía escalar es entropy_map.sum()
        entropy_map = -posterior * np.log(posterior + 1e-12)
        
        # EIG map — Eq. 6 del paper
        posterior_repeated = np.tile(posterior[:, :, np.newaxis, np.newaxis], (1, 1, *self.grid.size())) # Igual que en código Gonza
        expected_ig_map = 0.5 * np.sum(posterior_repeated * self.visibility_map.fovea_map, axis=(0, 1))

        if image_name not in self._maps_buffer:
            self._maps_buffer[image_name] = {
                'visual_evidence': [],
                'posterior':       [],
                'entropy_map':     [],
                'expected_ig_map': [],   
                'fixations_y':     [],
                'fixations_x':     [],
            }

        buf = self._maps_buffer[image_name]
        buf['visual_evidence'].append(visual_evidence.copy())
        buf['posterior'].append(posterior.copy())
        buf['entropy_map'].append(entropy_map.copy())
        buf['expected_ig_map'].append(expected_ig_map.copy())   
        buf['fixations_y'].append(int(current_fixation[0]))
        buf['fixations_x'].append(int(current_fixation[1]))

    def save_scanpaths(self, scanpaths):
        """Guarda los mapas acumulados como archivos .npz, uno por imagen."""
        makedirs(self.maps_dir, exist_ok=True)

        for image_name, buf in self._maps_buffer.items():
            human_sp = self.human_scanpaths.get(image_name, {})
            out_file = path.join(self.maps_dir, image_name[:-4] + '.npz')

            np.savez_compressed(
                out_file,
                visual_evidence=np.array(buf['visual_evidence'], dtype=np.float32),
                posterior=np.array(buf['posterior'],             dtype=np.float32),
                entropy_map=np.array(buf['entropy_map'],         dtype=np.float32),
                expected_ig_map=np.array(buf['expected_ig_map'], dtype=np.float32), 
                fixations_y=np.array(buf['fixations_y'],         dtype=np.int16),
                fixations_x=np.array(buf['fixations_x'],         dtype=np.int16),
                memory_set=np.array(human_sp.get('memory_set', []), dtype=object),
                target_stim=str(human_sp.get('target_stim', '')),
                target_found=bool(human_sp.get('target_found', False)),
            )

            print(f'  Guardado: {out_file}  ({len(buf["entropy_map"])} fijaciones)')

        self._maps_buffer = {}


def run_subject(dataset_name, config_name, subject_id, filters_mss):
    dataset_path = path.join(model_constants.DATASETS_PATH, dataset_name)
    trials_file = path.join(dataset_path, 'trials_properties.json')

    dataset_info = loader.load_dataset_info(dataset_path)
    human_scanpaths = loader.load_human_scanpaths(dataset_info['scanpaths_dir'], subject_id)

    if not human_scanpaths:
        print(f'  No se encontraron scanpaths para {subject_id}. Saltando.')
        return

    config = loader.load_config(
        model_constants.CONFIG_DIR, config_name,
        model_constants.IMAGE_SIZE, dataset_info['max_scanpath_length'],
        'all', human_scanpaths, follow_human_scanpath=True
    )

    trials_properties = loader.load_trials_properties(
        trials_file, None, None, human_scanpaths, checkpoint=None
    )

    output_path = path.join(
        'Results', f'{dataset_name}_dataset', config_name,
        'human_maps', subject_id
    )
    makedirs(output_path, exist_ok=True)  # run() usa mkdir y falla si el padre no existe

    extractor = HumanMapExtractor(
        config, dataset_info, trials_properties, output_path,
        human_scanpaths, model_constants.SIGMA, filters_mss, subject_id
    )
    extractor.run()


def main(dataset_name, config_name, subject_id=None, filters_mss=[]):
    dataset_path = path.join(model_constants.DATASETS_PATH, dataset_name)
    dataset_info = loader.load_dataset_info(dataset_path)
    scanpaths_dir = dataset_info['scanpaths_dir']

    if subject_id is not None:
        subjects = [subject_id]
    else:
        subjects = sorted([
            f.replace('_scanpaths.json', '')
            for f in listdir(scanpaths_dir)
            if f.endswith('_scanpaths.json')
        ])

    print(f'Dataset: {dataset_name} | Config: {config_name} | Sujetos: {len(subjects)}')
    for i, subj in enumerate(subjects, 1):
        print(f'\n[{i}/{len(subjects)}] Sujeto: {subj}')
        run_subject(dataset_name, config_name, subj, filters_mss)

    print(f'\nMapas guardados en: {MAPS_OUTPUT_DIR}/{dataset_name}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extrae mapas de evidencia visual, posterior y entropía en fijaciones humanas'
    )
    parser.add_argument('--d', '--dataset', type=str, default='HSEM',
                        help='Nombre del dataset (ej: HSEM)')
    parser.add_argument('--cfg', '--config', type=str, default='elm_final',
                        help='Nombre de la config (ej: elm_final)')
    parser.add_argument('--s', '--subject', type=str, default=None,
                        help='ID del sujeto (ej: et_117969). Sin valor = todos los sujetos.')
    args = parser.parse_args()
    main(args.d, args.cfg, args.s)
