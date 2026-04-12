from Metrics.scripts import human_scanpath_prediction
from .utils import utils
from .prior import prior
import numpy as np
import time
import importlib
#from os import path
from .visibility_map import VisibilityMap
from .grid import Grid
import sys
from os import mkdir, path
import time
from .visual_evidence_history import NoHistoryVisualEvidence, SimpleHistoryVisualEvidence, DegradedHistoryVisualEvidence
from .target_absent import TargetAbsentStrategy

class VisualSearcher:
    _capture_fixation_maps = False  # Activated only by HumanMapExtractor to extract per-fixation maps

    def __init__(self, config, dataset_info, trials_properties, output_path, sigma,filters_mss):
        " Creates a new instance of the visual search model "
        """ Input:
                Config (dict). One entry. Fields:
                    search_model      (string)   : bayesian, greedy
                    target_similarity (string)   : correlation, geisler, ssim, ivsn
                    prior             (string)   : deepgaze, mlnet, flat, center
                    max_saccades      (int)      : maximum number of saccades allowed
                    cell_size         (int)      : size (in pixels) of the cells in the grid
                    scale_factor      (int)      : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                    additive_shift    (int)      : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                    save_probability_maps (bool) : indicates whether to save the probability map to a file after each saccade or not
                    proc_number       (int)      : number of processes on which to execute bayesian search
                    image_size        (int, int) : image size on which the model will operate
                    save_similarity_maps (bool)  : indicates whether to save the target similarity map for each image in bayesian search
                Dataset info (dict). One entry. Fields:
                    name          (string)         : name of the dataset
                    images_dir    (string)         : folder path where search images are stored
                    targets_dir   (string)         : folder path where the targets are stored
                    saliency_dir  (string)         : folder path where the saliency maps are stored
                    target_similarity_dir (string) : folder path where the target similarity maps are stored
                    image_height  (int)            : default image height (in pixels)
                    image_width   (int)            : default image width (in pixels)
                Trials properties (dict):
                    Each entry specifies the data of the image on which to run the visual search model. Fields:
                    image  (string)               : image name (where to look)
                    target (string)               : name of the target image (what to look for)
                    target_matched_row (int)      : starting Y coordinate, in pixels, of the target in the image
                    target_matched_column (int)   : starting X coordinate, in pixels, of the target in the image
                    target_height (int)           : height of the target in pixels
                    target_width (int)            : width of the target in pixels
                    initial_fixation_row (int)    : row of the first fixation on the image
                    initial_fixation_column (int) : column of the first fixation on the image
                Output path     (string)        : folder path where scanpaths and probability maps will be stored
        """     
        self.cell_size  = config['cell_size']
        self.config = config
        self.dataset_name = dataset_info['dataset_name']
        self.model_image_size = config['image_size']
        self.max_saccades             = config['max_saccades']
        self.grid                     = Grid(np.array(self.model_image_size), self.cell_size)
        self.scale_factor             = config['scale_factor']
        self.additive_shift           = config['additive_shift']
        self.seed                     = config['seed']
        self.save_probability_maps    = config['save_probability_maps']
        self.save_similarity_maps     = config['save_similarity_maps']
        self.number_of_processes      = config['proc_number']
        self.visibility_map           = VisibilityMap(self.model_image_size, self.grid, sigma,config["fovea_exponent"],config["peripheral_exponent"])
        
        self.search_model             = self.initialize_model(config['search_model'], config['norm_cdf_tolerance'])
        self.target_similarity_dir    = dataset_info['target_similarity_dir']
        self.target_similarity_method = config['target_similarity']
        self.output_path              = output_path

        self.history_size             = config['history_size']
        self.trials_properties        = trials_properties
        self.images_dir            = dataset_info['images_dir']
        self.targets_dir           = dataset_info['targets_dir']
        self.saliency_dir          = dataset_info['saliency_dir']
        self.target_similarity_dir = dataset_info['target_similarity_dir']
        self.prior_name    = config['prior']        
        target_selectors = ["CorrectTarget","MinEntropy","Random","LikelihoodMean","MinEntropy2D"]
        target_present_conditions = ["Oracle","ConsecutiveFix","NeuralNet"]
        if ('target_present_condition' in config.keys()) and (config['target_present_condition'] in target_present_conditions):
            target_present_condition_class = config['target_present_condition']
        else:
            target_present_condition_class = "Oracle"
        self.target_present_condition = getattr(importlib.import_module('.target_present_condition', 'Model.visualsearch'), target_present_condition_class)()
        
        self.prior_as_fixation = ('prior_as_fixation' in config.keys()) and config["prior_as_fixation"]
        self.history_degradation = ('history_degradation' in config.keys()) and config["history_degradation"]
        self.target_selector_degradation = ('target_selector_degradation' in config.keys()) and config["target_selector_degradation"]
        self.target_selector_alpha = config["target_selector_alpha"] if ('target_selector_alpha' in config.keys()) else 0.6
        self.fovea_filter = ('fovea_filter' in config.keys()) and config["fovea_filter"]
        self.cell_size  = config['cell_size']

        if ('target_index_selector' in config.keys()) and (config['target_index_selector'] in target_selectors):
            self.target_index_selector_name = config['target_index_selector']
        else:
            self.target_index_selector_name = "MinEntropy"
        self.target_selector_class = getattr(importlib.import_module('.target_selector', 'Model.visualsearch'), self.target_index_selector_name)
        self.target_absent_strategy = TargetAbsentStrategy()
        self.posterior_heatmap_directory = path.join('Heatmaps','Posteriors',self.config["name"]) 
        self.elm_heatmap_directory = path.join('Heatmaps','Expected_info_gain',f'{self.config["name"]}')
        self.filters_mss = filters_mss

    def save_scanpaths(self,scanpaths):
        utils.save_scanpaths(self.output_path, scanpaths)

    def should_skip(self,image_name):
        return False

    def run(self):
        """
            Output:
                Output_path/scanpaths/Scanpaths.json: Dictionary indexed by image name where each entry contains the scanpath for that given image, alongside the configuration used.
                Output_path/probability_maps/: In this folder, the probability map computed for each saccade is stored. This is done for every image in trials_properties. (Only if save_probability_maps is true.)
                Output_path/similarity_maps/: In this folder, the target similarity map computed for each image is stored. This is done for every image in trials_properties. (Only if save_similarity_maps is true.)
        """
        
        print('Press Ctrl + C to interrupt execution and save a checkpoint \n')

        # If resuming execution, load previously generated data
        if not path.exists(self.output_path):
            mkdir(path.abspath(self.output_path))
        scanpaths, targets_found, previous_time = utils.load_data_from_checkpoint(self.output_path)

        trial_number = len(scanpaths)
        total_trials = len(self.trials_properties) + trial_number
        start = time.time()

        try:
            for trial in self.trials_properties:
                trial_number += 1
                image_name  = trial['image']
                target_name = trial['target']    
                if not ('memory_set' in trial):
                    trial['memory_set'] = [target_name]
                
                if len(trial['memory_set']) in self.filters_mss:
                    continue
            
                print('Searching in image ' + image_name + ' (' + str(trial_number) + '/' + str(total_trials) + ')...')
                
                image       = utils.load_image(self.images_dir, image_name, self.model_image_size)
                self.image_size = (trial["image_height"],trial["image_width"])
                image_prior = prior.load(image, image_name, self.model_image_size, self.prior_name, self.saliency_dir,target_name,path.join(self.images_dir, image_name))
                
                initial_fixation = [
                    utils.rescale_coordinate(trial['initial_fixation_row'], self.image_size[0], self.model_image_size[0]),
                    utils.rescale_coordinate(trial['initial_fixation_column'], self.image_size[1], self.model_image_size[1])
                ]
                target_bbox_in_grid, next_iteration = utils.get_target_bbox(trial,image_name,self.grid, self.image_size, self.model_image_size)
                if next_iteration or self.should_skip(image_name):
                    continue
                trial_scanpath = self.search(image_name, image, image_prior, trial['memory_set'],target_name, target_bbox_in_grid, initial_fixation)
                if trial_scanpath:
                    # If there were no errors, save the scanpath
                    utils.add_scanpath_to_dict(image_name, trial_scanpath, target_bbox_in_grid, trial['target_object'], self.grid, self.config, self.dataset_name, scanpaths,trial['memory_set'])
                    targets_found += trial_scanpath['target_found']
        except KeyboardInterrupt:
            time_elapsed = time.time() - start + previous_time
            utils.save_checkpoint(self.config, scanpaths, targets_found, self.trials_properties, time_elapsed, self.output_path)        
            sys.exit(0)

        time_elapsed = time.time() - start + previous_time
        self.save_scanpaths(scanpaths)

        utils.erase_checkpoint(self.output_path)

        print('Total targets found: ' + str(targets_found) + '/' + str(len(scanpaths)))
        print('Total time elapsed:  ' + str(round(time_elapsed, 4))   + ' seconds')
        # Agregado para tener un resultado intermedio
        return targets_found, len(scanpaths)

    
    def get_memory_set_names(self,memory_set_names,image_name):
        return memory_set_names

    def get_memset_weights(self,memory_set):
        return np.ones(len(memory_set))
    
    def get_first_fixation(self,initial_fixation):
        return self.grid.map_to_cell(initial_fixation)
    
    def get_current_fixation(self,fixation_number,fixations):
        return fixations[fixation_number]
    
    def plot_heatmap(self):
        return self.save_probability_maps
    
    def save_metrics(self,image_name):
        return

    def on_fixation_computed(self, fixation_number, current_fixation, visual_evidence, posterior, image_name):
        """Hook called after each fixation's maps are computed. No-op by default.
        Override in subclasses (e.g. HumanMapExtractor) by setting _capture_fixation_maps = True."""
        pass

    def search(self, image_name, image, image_prior, memory_set_names,target_name,target_bbox, initial_fixation):
        " Given an image, a target, and a prior of that image, it looks for the object in the image, generating a scanpath "
        """ Input:
            Specifies the data of the image on which to run the visual search model. Fields:
                image_name (string)         : name of the image
                image (2D array)            : search image
                image_prior (2D array)      : grayscale image with values between 0 and 1 that serves as prior
                memory_set_names (List)     : filenames of the stimuli
                target_bbox (array)         : bounding box (upper left row, upper left column, lower right row, lower right column) of the target inside the search image
                initial_fixation (int, int) : row and column of the first fixation on the search image
            Output:
                image_scanpath   (dict)      : scanpath made by the model on the search image, alongside a 'target_found' field which indicates if the target was found
                probability_maps (csv files) : if self.save_probability_maps is True, the probability map for each saccade is stored in a .csv file inside a folder in self.output_path 
                similarity_maps  (png files) : if self.save_similarity_maps is True, the target similarity map for each image is stored inside a folder in self.output_path
        """
        # Convert prior to grid
        image_prior = self.grid.reduce(image_prior, mode='mean')
        grid_size   = self.grid.size()

        # Check prior dimensions
        if not(image_prior.shape == grid_size):
            print(image_name + ': prior image\'s dimensions don\'t match dataset\'s dimensions')
            return {}
        # Sum probabilities
        image_prior = prior.sum(image_prior)

        memory_set_names = self.get_memory_set_names(memory_set_names,image_name)

        # Initialize fixations matrix
        fixations = np.empty(shape=(self.max_saccades + 1, 2), dtype=int)
        fixations[0] = self.get_first_fixation(initial_fixation)
        if not(utils.are_within_boundaries(fixations[0], fixations[0], np.zeros(2), grid_size)):
            print(image_name + ': initial fixation falls off the grid')
            return {}

        # Initialize working memory variables
        memory_set = list(map(lambda x: utils.load_image(self.targets_dir,x),memory_set_names))
        visual_evidences = np.array(list(map(lambda i: self.initialize_target_similarity_map(memory_set[i], image,target_bbox, image_name,memory_set_names[i],target_name,len(memory_set)),range(0,len(memory_set)))))
        memset_weights = self.get_memset_weights(memory_set)
        memset_weights = memset_weights/np.sum(memset_weights)
        target_selector = self.target_selector_class(memset_weights,memory_set_names,target_name)
        
        # Search
        print('Fixation:', end=' ')
        target_found = False
        start = time.time()
        searched_object_indexes_one_hot = np.zeros((self.max_saccades, len(memory_set)))
        info_gained = []

        if (self.history_size != None):            
            if self.history_degradation and (self.history_size > 0):
                visual_evidence_history_factory = DegradedHistoryVisualEvidence(self.prior_as_fixation, self.max_saccades, grid_size, image_prior,self.history_size)                    
            elif (self.history_size < self.max_saccades):
                visual_evidence_history_factory = SimpleHistoryVisualEvidence(self.prior_as_fixation, self.max_saccades, grid_size, image_prior,self.history_size)
            else:
                visual_evidence_history_factory = NoHistoryVisualEvidence(self.prior_as_fixation, self.max_saccades, grid_size, image_prior,self.history_size)
        else:
            visual_evidence_history_factory = NoHistoryVisualEvidence(self.prior_as_fixation, self.max_saccades, grid_size, image_prior,self.history_size)
        history_visual_evidence = visual_evidence_history_factory.get_history_visual_evidence()    
                   
        for fixation_number in range(self.max_saccades + 1):
            current_fixation = self.get_current_fixation(fixation_number,fixations)
            print(fixation_number + 1, end=' ')
            # If the limit has been reached, don't compute the next fixation            
            if fixation_number == self.max_saccades:
                break
            #TODO: Target Absent
            if self.target_absent_strategy.should_break(fixation_number, history_visual_evidence):
                fixations = fixations[:fixation_number]
                break
            # Compute the unnormalized posteriors for each object in the memory set, according to the target selector
            posteriors_unnormalized = target_selector.get_posteriors_unnormalized(visual_evidences,image_prior,current_fixation,fixation_number,visual_evidence_history_factory)
            # Select the next object to search for
            selected_posterior_index = target_selector.select(posteriors_unnormalized)
            # Apply the target selector degradation if applicable
            searched_object_indexes_one_hot, selected_posterior_index = target_selector_degradation(selected_posterior_index,searched_object_indexes_one_hot,fixation_number,
                                                                                        self.target_selector_degradation,self.target_selector_alpha)

            #Update working memory with the visual evidence of the selected object
            visual_evidence_at_fixation = visual_evidences[selected_posterior_index].at_fixation(current_fixation, fixation_number)
            visual_evidence_history_factory.update_values(visual_evidence_at_fixation, fixation_number)
            likelihood_times_prior = posteriors_unnormalized[selected_posterior_index]

            # Compute the posterior
            posterior = likelihood_times_prior / np.sum(likelihood_times_prior)
            if self._capture_fixation_maps:  # Only active in HumanMapExtractor
                self.on_fixation_computed(fixation_number, current_fixation, visual_evidence_at_fixation, posterior, image_name)

            # Compute next fixation
            next_fix_x,next_fix_y,info = self.search_model.next_fixation(posterior, image_name, fixation_number, self.output_path,path.join(self.elm_heatmap_directory,f'MSS {len(memory_set)}',f'{image_name[:-4]}'),current_fixation)            
            info_gained.append(info)
            next_fix = (next_fix_x,next_fix_y)

            # Save the posterior heatmap if applicable
            if self.plot_heatmap():
                utils.save_csv_heatmap(path.join(self.posterior_heatmap_directory,f'MSS {len(memory_set)}',f'{image_name[:-4]}'),f'{fixation_number}_{current_fixation[0]}_{current_fixation[1]}.csv',posterior)
            
            # Check if the target is present
            if self.target_present_condition.end_trial(target_bbox, current_fixation, fixation_number, fixations):               
                if not target_bbox is None and utils.are_within_boundaries(current_fixation, current_fixation, (target_bbox[0], target_bbox[1]), (target_bbox[2] + 1, target_bbox[3] + 1)):
                    target_found = True
                fixations = fixations[:fixation_number + 1]
                break                   
            fixations[fixation_number + 1] =  next_fix

        end = time.time()

        if target_found:
            print('\nTarget found!')
        else:
            print('\nTarget NOT FOUND!')
        print('Time elapsed: ' + str(end - start) + '\n')

        # Note: each x coordinate refers to a column in the image, and each y coordinate refers to a row in the image
        scanpath_x_coordinates = utils.get_coordinates(fixations, axis=1)
        scanpath_y_coordinates = utils.get_coordinates(fixations, axis=0)

        self.save_metrics(image_name)
        searched_object_indexes = np.array(searched_object_indexes_one_hot[:fixation_number]).argmax(axis=1)

        return { 'elm_info_gained':info_gained,'searched_object_indexes':searched_object_indexes,'target_found' : target_found, 'scanpath_x' : scanpath_x_coordinates, 'scanpath_y' : scanpath_y_coordinates }

    def initialize_model(self, search_model, norm_cdf_tolerance):
        module = importlib.import_module('.models.'+ search_model +"_model", 'Model.visualsearch')
        searcher_class = getattr(module, search_model.capitalize() +"Model")
        return searcher_class(self.grid.size(), self.visibility_map, norm_cdf_tolerance, self.number_of_processes, self.save_probability_maps,self.plot_heatmap())

    def initialize_target_similarity_map(self,  target, image, target_bbox, image_name,stim_name,target_name,mss):
        # Load corresponding module, which has the same name in lower case
        module = importlib.import_module('.target_similarity.' + self.target_similarity_method.lower(), 'Model.visualsearch')
        # Get the class
        target_similarity_class = getattr(module, self.target_similarity_method.capitalize())
        target_similarity_map   = target_similarity_class(image_name, stim_name,target_name, image, target, target_bbox, self.visibility_map, self.scale_factor, self.additive_shift, self.grid, self.seed, \
            self.number_of_processes, self.save_similarity_maps, self.target_similarity_dir,path.join('Heatmaps','Visual Evidences',f'{self.config["name"]}',f'MSS {mss}',f'{image_name[:-4]}'),self.fovea_filter)
        return target_similarity_map

def target_selector_degradation(selected_posterior_index,searched_object_indexes_one_hot,fixation_number,degradation_flag,alpha):
    searched_object_indexes_one_hot[fixation_number, selected_posterior_index] = 1
    # Exponentially weighted average of the selected posteriors
    if degradation_flag and fixation_number > 0:
        exponential_moving_average_previous = searched_object_indexes_one_hot[fixation_number - 1]
        selector_weight_sum = alpha * exponential_moving_average_previous + (1 - alpha) * searched_object_indexes_one_hot[fixation_number]

        searched_object_indexes_one_hot[fixation_number] = selector_weight_sum
        selected_posterior_index = np.argmax(selector_weight_sum)
    return searched_object_indexes_one_hot, selected_posterior_index

class VisualSearcherSubject(VisualSearcher):
    def __init__(self, config, dataset_info, trials_properties, output_path, human_scanpaths,sigma,filters_mss,follow_human_scanpath=False):
        self.follow_human_scanpath = follow_human_scanpath
        super().__init__(config, dataset_info, trials_properties, output_path,sigma,filters_mss)
        if not human_scanpaths:
            raise ValueError('Human scanpaths must be provided')
        self.human_scanpaths = human_scanpaths
        if ('primacy_and_recency' in config.keys()) and config["primacy_and_recency"]:
            self.primacy_and_recency = True
        else:
            self.primacy_and_recency = False
        if ('seen_targets_weights' in config.keys()) and config["seen_targets_weights"]:
            self.seen_targets_weights = True
        else:
            self.seen_targets_weights = False
        utils.rescale_scanpaths(self.grid, self.human_scanpaths)

    def save_scanpaths(self,scanpaths):
        if self.follow_human_scanpath:
            filename = 'Subject_scanpaths.json'
            utils.save_scanpaths(self.output_path, self.human_scanpaths, filename=filename)
        else:
            filename = "Subject_scanpaths_only_memset.json"
            utils.save_scanpaths(self.output_path, scanpaths, filename=filename)

    def should_skip(self,image_name):
        if utils.exists_probability_maps_for_image(image_name, self.output_path):
            print('Loaded previously computed probability maps for image ' + image_name)
            human_scanpath_prediction.save_scanpath_prediction_metrics(self.human_scanpaths[image_name], image_name, self.output_path)
            return True
        return False

    def get_memory_set_names(self, memory_set_names,image_name):
        self.current_human_scanpath = self.human_scanpaths[image_name]
        if self.follow_human_scanpath:
            self.max_saccades = len(self.current_human_scanpath['Y']) - 1
        memory_set_names = self.current_human_scanpath.get("memory_set",memory_set_names)
        return memory_set_names
    
    def get_first_fixation(self, initial_fixation):
        if self.follow_human_scanpath:
            current_human_fixations = np.array(list(zip(self.current_human_scanpath['Y'], self.current_human_scanpath['X'])))
            return current_human_fixations[0]
        else:
            return self.grid.map_to_cell(initial_fixation)

    def get_memset_weights(self,memory_set):
        memset_weights = np.ones(len(memory_set))
        if self.primacy_and_recency:
            seen_order = self.current_human_scanpath["seen_order"]
            if len(seen_order) > 0:
                memset_weights = np.zeros(len(memory_set)) # Set to 0 the weights of the unseen objects
                memset_weights[seen_order] = 1 # Set to 1 the weights of the seen objects
                memset_weights[seen_order[-1]] = 2 # Set to 2 the weight of the last seen object
                memset_weights[seen_order[0]] = 2 # Set to 2 the weight of the first seen object
        if self.seen_targets_weights:
            seen_order = self.current_human_scanpath["seen_order"]
            targets_weights = self.current_human_scanpath["fix_proportion_mem"]
            if len(seen_order) > 0:
                memset_weights = np.zeros(len(memory_set)) # Set to 0 the weights of the unseen objects
                for index,value in enumerate(seen_order):
                    memset_weights[value] = targets_weights[index]
        return memset_weights
    
    def plot_heatmap(self):
        # return not self.follow_human_scanpath and self.save_probability_maps
        return self.save_probability_maps

    def get_current_fixation(self, fixation_number, fixations):        
        if self.follow_human_scanpath:
            current_human_fixations = np.array(list(zip(self.current_human_scanpath['Y'], self.current_human_scanpath['X'])))
            current_fixation = current_human_fixations[fixation_number]
        else:
            current_fixation = fixations[fixation_number]
        return current_fixation
    
    def save_metrics(self,image_name):
        if self.follow_human_scanpath:
            human_scanpath_prediction.save_scanpath_prediction_metrics(self.current_human_scanpath, image_name, self.output_path)