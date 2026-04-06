from . import constants
import argparse

from .scripts import utils
from os import path,listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def _get_colors(n):
    cmap1 = plt.get_cmap('tab20')  # 20 colors
    cmap2 = plt.get_cmap('Set3')   # 12 colors
    colors = [cmap1(i % 20) for i in range(min(n, 20))]  # Take from tab20
    if n > 20:
        colors += [cmap2(i % 12) for i in range(n - 20)]  # Fill the remaining from Set3
    return colors

def main(datasets, models,metrics,results_folder,filters_mss = {}):
    # Add the models that are not in the filters_mss dictionary as keys with a value of []
    for model in models:
        if model not in filters_mss:
            filters_mss[model] = []
    all_filters_mss = list(map(lambda x: filters_mss[x], filters_mss.keys()))
    # Concatenate all the filters_mss lists into one set
    all_filters_mss = set().intersection(*all_filters_mss)
    filters_mss["Humans"] = all_filters_mss
    models = sorted(list(set(models)))
    datasets = list(set(datasets))    

    datasets_results = {}
    colors = _get_colors(len(models))    
    colors_dict = dict(zip(models,colors))
    colors_dict['Humans'] = constants.HUMANS_COLOR

    for dataset_name in datasets:
        dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
        dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))

        human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])
        dataset_results_dir = path.join(results_folder, dataset_name + '_dataset')
        if not path.isdir(dataset_results_dir):
            print('No results found for ' + dataset_name + ' dataset')
            continue
        max_scanpath_length = dataset_info['max_scanpath_length']
        temp_list = []
        for model_name in models:
            if not path.isdir(path.join(dataset_results_dir, model_name)):
                print('No results found for ' + model_name + ' in ' + dataset_name + ' dataset')
                continue
            temp_list.append({'Scanpath_file':'Scanpaths.json','Model':model_name})

        for human_scanpath_file in listdir(human_scanpaths_dir):
            temp_list.append({'Scanpath_file':human_scanpath_file, 'Model':'Humans'})
        scanpath_metadata = pd.DataFrame(temp_list)
        for metric in metrics:
            constants.NAME_METRIC_MAP[metric](dataset_name, scanpath_metadata, dataset_results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict).compute()

        dataset_results = utils.load_dict_from_json(path.join(dataset_results_dir, constants.FILENAME))
        if dataset_results != {}:
            datasets_results[dataset_name] = dataset_results
            dataset_results_table = utils.create_table(dataset_results)
            utils.plot_table(dataset_results_table, title=dataset_name + ' dataset', save_path=dataset_results_dir, filename='Table.png')
            #utils.rose_plot(dataset_results,colors_dict,save_path=dataset_results_dir,filename='Rose.png')
            utils.scores_plots(dataset_results,colors_dict,save_path=dataset_results_dir,filename='Scores.png')
            # Plot the labels and legends in a separate plot
            utils.plot_legends_and_labels(dataset_results,colors_dict,save_path=dataset_results_dir,filename='Labels_and_Legends.png')

    if "hsp" in metrics and ("perf" in metrics or "perfs" in metrics) and ("mm" in metrics or "mms" in metrics):
        final_table = utils.average_results(datasets_results, save_path=results_folder, filename='Scores.json')
        # cambiar el average results por algo que trabaje con dataframes
        utils.plot_table(final_table, title='Ranking', save_path=results_folder, filename='Ranking.png')

