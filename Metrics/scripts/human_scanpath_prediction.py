from . import utils
from .. import constants
from os import path, listdir, pardir, remove
import pandas as pd
import numpy as np
import shutil
import numba
import importlib
from concurrent.futures import ProcessPoolExecutor

""" Computes Human Scanpath Prediction on the visual search models for a given dataset. 
    See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information.
    The methods for computing AUC and NSS were taken from https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py
"""

class HumanScanpathPrediction:
    def __init__(self, dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict):
        self.dataset_name = dataset_name
        self.scanpath_metadata = scanpath_metadata
        self.dataset_results_dir = results_dir
        self.human_scanpaths_dir = human_scanpaths_dir
        self.filters_mss = filters_mss
        self.color_model_mapping = colors_dict
        self.general_results_dir = path.dirname(self.dataset_results_dir)


    def compute(self):
        baseline_models_results = self.add_baseline_models()
        unique_models = self.scanpath_metadata[self.scanpath_metadata['Model'] != "Humans"]['Model'].unique()
        model_results = pd.concat(list(map(lambda x: self.compute_metrics_for_model(x), unique_models)), axis=0)
        # Place the "Model" column in model_results at the beginning of the dataframe
        model_results = model_results[['Model','MSS','AUC','NSS','IG','LL']]
        results = pd.concat([baseline_models_results, model_results], axis=0)
        results = results.round(3)
        self.save_results(constants.FILENAME, results)

    def add_baseline_models(self):
        baseline_filepath = path.join(self.dataset_results_dir, 'baseline_hsp.json')
        baseline_csvpath = path.join(self.dataset_results_dir, 'baseline_hsp.csv')
        baseline_averages = utils.load_dict_from_json(baseline_filepath)
        if baseline_averages:
            list = []
            for model in baseline_averages:
                for mss in baseline_averages[model]:
                    for image in baseline_averages[model][mss]:
                        list.append(pd.Series({"Model":model,"MSS":mss,"Image":image,"AUC":baseline_averages[model][mss][image]['AUC'],"NSS":baseline_averages[model][mss][image]['NSS'],"IG":baseline_averages[model][mss][image]['IG'],"LL":baseline_averages[model][mss][image]['LL']}))
            baseline_averages = pd.DataFrame(list)
            pd.DataFrame.to_csv(baseline_averages,path.join(self.dataset_results_dir, 'baseline_hsp.csv'),index=False)
            remove(baseline_filepath)
        elif path.isfile(baseline_csvpath):
            baseline_averages = pd.read_csv(baseline_csvpath)
        else:
            human_metadata = self.scanpath_metadata[self.scanpath_metadata['Model'] == 'Humans']
            baseline_averages = self.run_baseline_models(human_metadata)
            pd.DataFrame.to_csv(baseline_averages,path.join(self.dataset_results_dir, 'baseline_hsp.csv'),index=False)       
        return baseline_averages.drop("Image",axis=1).groupby(["Model","MSS"]).mean().reset_index()

    def run_baseline_models(self, human_metadata):
        """ Compute every metric for center bias, uniform and gold standard models in the given dataset """
        dataset_path = path.join(constants.DATASETS_PATH, self.dataset_name)
        dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))
        image_size   = (dataset_info['image_height'], dataset_info['image_width'])
        baseline_results = human_metadata.apply(lambda row: self.compute_baseline_metrics_for_subject(row, image_size,center_bias(shape=image_size),uniform(shape=image_size)), axis=1)
        baseline_averages = pd.concat(list(baseline_results), axis=0).reset_index(drop=True).groupby(['Model','MSS','Image']).mean().reset_index()
        return baseline_averages

    def compute_baseline_metrics_for_subject(self, row, image_size, center_bias, uniform):
        subject_scanpaths_file = row['Scanpath_file']
        subject_scanpaths_path = path.join(self.human_scanpaths_dir, subject_scanpaths_file)
        subject = subject_scanpaths_file[:-15]
        print('[Human Scanpath Prediction] Running baseline models on ' + self.dataset_name + ' dataset using ' + subject + ' scanpaths')
        subject_scanpaths = utils.load_dict_from_json(subject_scanpaths_path)

        useful_images = list(filter(lambda x: not len(subject_scanpaths[x]['memory_set']) in self.filters_mss["Humans"] if 'memory_set' in subject_scanpaths[x] else True, subject_scanpaths.keys()))
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.compute_baseline_metrics_for_image, image, subject_scanpaths[image], image_size, center_bias, uniform, subject_scanpaths_file) for image in useful_images]
            baselines_for_image = [future.result() for future in futures]       
        return pd.concat(baselines_for_image, axis=0).reset_index(drop=True)

    def compute_baseline_metrics_for_image(self,image_name, subject_scanpath, image_size, center_bias, uniform, subject):
        gold_standard_model = gold_standard(image_name, image_size, self.human_scanpaths_dir, excluded_subject=subject)
        scanpath_x = [int(x) for x in subject_scanpath['X']]
        scanpath_y = [int(y) for y in subject_scanpath['Y']]
        baseline_models = {"center_bias": center_bias, "uniform": uniform, "gold_standard": gold_standard_model}
        results = []
        mss = len(subject_scanpath['memory_set']) if 'memory_set' in subject_scanpath else 1
        results = pd.DataFrame(list(map(lambda x: compute_trial_metrics(len(scanpath_x), scanpath_x, scanpath_y, None, x), baseline_models.values())), columns=['AUC','NSS','IG','LL'])
        results['Model'] = list(baseline_models.keys())
        results['MSS'] = mss
        results['Image'] = image_name
        return pd.DataFrame(results)

    def compute_metrics_for_model(self, model_name):
        model_output_path     = path.join(self.dataset_results_dir, model_name)    
        human_scanpaths_files = utils.sorted_alphanumeric(listdir(self.human_scanpaths_dir))
        model_average_file    = path.join(model_output_path, 'human_scanpath_prediction_mean_per_image.json')
        csv_file              = path.join(model_output_path, 'human_scanpath_prediction_mean_per_image.csv')
        average_results_per_image = utils.load_dict_from_json(model_average_file)
        if average_results_per_image:
            print('[Human Scanpath Prediction] Found previously computed results for ' + model_name)
            list = []
            for mss in average_results_per_image:
                for image in average_results_per_image[mss]:
                    list.append(pd.Series({"MSS":mss,"Image":image,"AUC":average_results_per_image[mss][image]['AUC'],"NSS":average_results_per_image[mss][image]['NSS'],"IG":average_results_per_image[mss][image]['IG'],"LL":average_results_per_image[mss][image]['LL']}))
            average_results_per_image = pd.DataFrame(list)
            average_results_per_image.to_csv(csv_file, index=False)
            remove(model_average_file)            
        elif path.isfile(csv_file):
            print('[Human Scanpath Prediction] Found previously computed results for ' + model_name)
            average_results_per_image = pd.read_csv(csv_file)
        else:
            model_filters_mss = self.filters_mss.get(model_name, [])
            for subject in human_scanpaths_files:             
                subject_name = subject[:-15]
                if not self.subject_already_processed(subject, subject_name, model_output_path):
                    model = importlib.import_module('Model.main')
                    print('[Human Scanpath Prediction] Running ' + model_name + ' on ' + self.dataset_name + ' dataset using subject ' + subject_name + ' scanpaths')
                    model.main(self.dataset_name,model_name, subject_name,filters_mss=model_filters_mss,results_folder = self.general_results_dir,follow_human_scanpath = True)            
            average_results_per_image = self.get_model_average_per_image(model_output_path)
            average_results_per_image.to_csv(csv_file, index=False)
        model_mean = average_results_per_image.drop("Image",axis=1).groupby(["MSS"]).mean().reset_index()
        model_mean['Model'] = model_name
        return model_mean

    def subject_already_processed(self, subject_file, subject_number, model_output_path):
        subjects_predictions_path  = path.join(model_output_path, 'subjects_predictions')
        subject_scanpath_file      = path.join(self.human_scanpaths_dir, subject_file)
        subject_predictions_file   = path.join(subjects_predictions_path, 'subject_' + subject_number + '_results.csv')
        subject_predictions_file_json   = path.join(subjects_predictions_path, 'subject_' + subject_number + '_results.json')
        if utils.is_contained_in(subject_scanpath_file, subject_predictions_file,"Image") or path.exists(subject_predictions_file_json):
            print('[Human Scanpath Prediction] Found previously computed results for subject ' + subject_number)
            return True        
        return False
    
    def get_model_average_per_image(self, model_output_path):
        subjects_results_path  = path.join(model_output_path, 'subjects_predictions')
        json_files = utils.list_files_ending(subjects_results_path,".json")
        # Turn the json files into pandas dataframes in order to save them as csv files
        for file in json_files:
            json_file = utils.load_dict_from_json(path.join(subjects_results_path, file))
            temp_list = []
            for mss in json_file:
                for image in json_file[mss]:
                    temp_list.append(pd.Series({"MSS":mss,"Image":image,"AUC":json_file[mss][image]['AUC'],"NSS":json_file[mss][image]['NSS'],"IG":json_file[mss][image]['IG'],"LL":json_file[mss][image]['LL']}))
            pd.DataFrame(temp_list).to_csv(path.join(subjects_results_path,file[:-5] + '.csv'),index=False)
            remove(path.join(subjects_results_path, file))
        subjects_results_files = utils.list_files_ending(subjects_results_path,".csv")
        df = pd.concat(list(map(lambda x: pd.read_csv(path.join(subjects_results_path,x)), subjects_results_files)))
        average_per_image = df.groupby(["MSS", "Image"]).mean().reset_index()
        return average_per_image
        
    def save_results(self, filename,hsp_df):
        dataset_metrics_file = path.join(self.dataset_results_dir, filename)
        dataset_metrics      = utils.load_dict_from_json(dataset_metrics_file)    
        hsp_df.apply(lambda x: self.save_model_results(dataset_metrics, x), axis=1)
        utils.save_to_json(dataset_metrics_file, dataset_metrics)
    
    def save_model_results(self, dataset_metrics, model_results):
        model = model_results['Model']
        mss = model_results['MSS']
        if model not in dataset_metrics:
            dataset_metrics[model] = {}
        utils.update_dict(dataset_metrics[model], "MSS "+str(mss),{"AUChsp":model_results['AUC'],"NSShsp":model_results['NSS'],"IGhsp":model_results['IG'],"LLhsp":model_results['LL']})
        return dataset_metrics

def save_scanpath_prediction_metrics(subject_scanpath, image_name, output_path):
    """ After creating the probability maps for each fixation in a given human subject's scanpath, visual search models call this method """
    probability_maps_path = path.join(output_path, 'probability_maps', image_name[:-4])
    if not path.exists(probability_maps_path):
        print('[Human Scanpath Prediction] No probability maps found for ' + image_name)
        return
    probability_maps = listdir(probability_maps_path)
    subject_fixations_x = np.array(subject_scanpath['X'], dtype=int)
    subject_fixations_y = np.array(subject_scanpath['Y'], dtype=int)
    mss = len(subject_scanpath['memory_set']) if 'memory_set' in subject_scanpath else 1
    trial_aucs, trial_nss, trial_igs, trial_lls = compute_trial_metrics(len(probability_maps) + 1, subject_fixations_x, subject_fixations_y, probability_maps_path)
    subject   = path.basename(output_path)
    file_path = path.join(output_path, pardir, subject + '_results.csv')
    if path.exists(file_path):
        model_subject_metrics = pd.read_csv(file_path)
    else:
        model_subject_metrics = pd.DataFrame(columns=['MSS', 'Image', 'AUC', 'NSS', 'IG', 'LL'])
    model_subject_metrics.loc[len(model_subject_metrics.index)] = [mss, image_name, np.mean(trial_aucs), np.mean(trial_nss), np.mean(trial_igs), np.mean(trial_lls)]
    model_subject_metrics.to_csv(file_path, index=False)
    # Clean up probability maps if their size is too big
    if utils.dir_is_too_heavy(probability_maps_path):
        shutil.rmtree(probability_maps_path)

def compute_trial_metrics(number_of_fixations, subject_fixations_x, subject_fixations_y, prob_maps_path, baseline_map=None):
    trial_aucs, trial_nss, trial_igs, trial_lls = [], [], [], []
    for index in range(1, number_of_fixations):
        if baseline_map is None:
            fixation_prob_map = pd.read_csv(path.join(prob_maps_path, 'fixation_' + str(index) + '.csv')).to_numpy()
        else:
            fixation_prob_map = baseline_map
        baseline_ig      = center_bias(fixation_prob_map.shape)
        baseline_ll      = uniform(fixation_prob_map.shape)
        auc, nss, ig, ll = compute_fixation_metrics(baseline_ig, baseline_ll, fixation_prob_map, subject_fixations_y[index], subject_fixations_x[index])
        trial_aucs.append(auc)
        trial_nss.append(nss)
        trial_igs.append(ig)
        trial_lls.append(ll)    
    return np.mean(trial_aucs), np.mean(trial_nss), np.mean(trial_igs), np.mean(trial_lls)

def compute_fixation_metrics(baseline_ig, baseline_ll, probability_map, human_fixation_y, human_fixation_x):
    auc = AUC(probability_map, human_fixation_y, human_fixation_x)
    nss = NSS(probability_map, human_fixation_y, human_fixation_x)
    ig  = infogain(probability_map, baseline_ig, human_fixation_y, human_fixation_x)
    ll  = infogain(probability_map, baseline_ll, human_fixation_y, human_fixation_x)
    return auc, nss, ig, ll

def uniform(shape):
    return np.ones(shape) / (shape[0] * shape[1])

def center_bias(shape):
    shape_dir = str(shape[0]) + 'x' + str(shape[1])
    filepath  = path.join(constants.CENTER_BIAS_PATH, shape_dir,  'center_bias.pkl')
    if path.exists(filepath):
        return utils.load_pickle(filepath)
    scanpaths_X, scanpaths_Y = utils.load_center_bias_fixations(model_size=shape)
    centerbias = utils.gaussian_kde(scanpaths_X, scanpaths_Y, shape)
    utils.save_to_pickle(centerbias, filepath)

    return centerbias

def gold_standard(image_name, image_size, subjects_scanpaths_path, excluded_subject):    
    scanpaths_X, scanpaths_Y = utils.aggregate_scanpaths(subjects_scanpaths_path, image_name, excluded_subject)
    if len(scanpaths_X) == 0:
        goldstandard_model = None
    else:
        goldstandard_model = utils.gaussian_kde(scanpaths_X, scanpaths_Y, image_size)
    return goldstandard_model

def NSS(probability_map, ground_truth_fixation_y, ground_truth_fixation_x, eps=2.2204e-20):
    """ The returned array has length equal to the number of fixations """
    mean  = np.mean(probability_map)
    std   = np.std(probability_map)
    value = np.copy(probability_map[ground_truth_fixation_y, ground_truth_fixation_x])
    value -= mean
    value = value if eps < value else 0.0
    if std:
        value /= std
    return value

def infogain(s_map, baseline_map, ground_truth_fixation_y, ground_truth_fixation_x):
    eps = 2.2204e-16
    s_map        = s_map / np.sum(s_map)
    baseline_map = baseline_map / np.sum(baseline_map)    
    return np.log2(eps + s_map[ground_truth_fixation_y, ground_truth_fixation_x]) - np.log2(eps + baseline_map[ground_truth_fixation_y, ground_truth_fixation_x])

def AUC(probability_map, ground_truth_fixation_y, ground_truth_fixation_x):
    """ Calculate AUC score for a given fixation """
    positive  = probability_map[ground_truth_fixation_y, ground_truth_fixation_x]
    negatives = probability_map.flatten()
    return auc_for_one_positive(positive, negatives)

def auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    return _auc_for_one_positive(positive, np.asarray(negatives))

@numba.jit(nopython=True)
def _auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    count = 0
    for negative in negatives:
        if negative < positive:
            count += 1
        elif negative == positive:
            count += 0.5
    return count / len(negatives)