import torch
import re
import os
from tqdm import tqdm

MODEL_FILE_NAME_REGEX = rf'_state_\d+_acc=[\d.eE+-]+_loss=[\d.eE+-]+\.pth$'
MODEL_FILE_NAME_REGEX_MATCH = rf'_state_(\d+)_acc=([\d.eE+-]+)_loss=([\d.eE+-]+)\.pth$'

########################################################################################
########################################################################################

def sorted_nicely(l): 
    """ 
    Sort the given iterable in the way that humans expect.
    https://stackoverflow.com/a/2669120/11814682
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)

########################################################################################
########################################################################################

def extract_metrics(file_name, exp_name=None):
    """
    Extract step, accuracy, and loss from the file name
    Args:
        file_name (str): The name of the file.
        exp_name (str): The name of the experiment.
    """
    pattern = '^' + ('(.*)' if exp_name is None else re.escape(exp_name)) + MODEL_FILE_NAME_REGEX_MATCH
    match_info = re.match(pattern, file_name)
    if match_info:
        i = 2 if exp_name is None else 1
        step = int(match_info.group(i))
        test_acc = float(match_info.group(i+1))
        test_loss = float(match_info.group(i+2))
        return step, test_acc, test_loss
    return None, None, None

########################################################################################
########################################################################################

def get_model_files(checkpoint_path, file_name=None):
    """
    Get the model files in the given directory.
    Args:
        checkpoint_path (str): The path to the directory containing the checkpoints.
        file_name (str): The name of the file.
    """
    pattern = f'^' + ('.*' if file_name is None else re.escape(file_name)) + MODEL_FILE_NAME_REGEX
    model_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)) and  re.match(pattern, f)]
    model_files = sorted_nicely(model_files)
    return model_files

########################################################################################
########################################################################################

def get_all_checkpoints(checkpoint_path, file_name, just_files=False):
    """
    Load all the checkpoints from the given directory.
    Args:
        checkpoint_path (str): The path to the directory containing the checkpoints.
        file_name (str): The name of the file.
        just_files (bool): If True, only the file paths will be returned. Otherwise, the models will be loaded.
    """

    model_files = get_model_files(checkpoint_path, file_name)
    # Extract metrics from the file names
    metrics_dict = {f: extract_metrics(f, file_name) for f in model_files} # {file : (step, test_acc, test_loss)}

    if file_name is not None :
        statistics = torch.load(os.path.join(checkpoint_path, f"{file_name}.pth"))
    else :
        statistics = None
    
    model_files = list(metrics_dict.keys())
    if just_files:
        return {metrics_dict[f][0] : os.path.join(checkpoint_path, f) for f in model_files}, statistics

    all_models = {metrics_dict[f][0] : torch.load(os.path.join(checkpoint_path, f), map_location='cpu') for f in tqdm(model_files)}

    return all_models, statistics


########################################################################################
########################################################################################

def get_all_checkpoints_per_trials(all_checkpoint_paths, file_name, just_files=False, verbose=False):
    
    all_models_per_trials = []
    all_statistics = []

    n_model = len(all_checkpoint_paths)
    for i, checkpoint_path in enumerate(all_checkpoint_paths) :
        print(f"Model {i+1}/{n_model}")

        if verbose : 
            print(checkpoint_path)
            #print(os.listdir(checkpoint_path))

        all_models, statistics = get_all_checkpoints(checkpoint_path, file_name, just_files)
        all_models_per_trials.append(all_models)
        all_statistics.append(statistics)
    
    #metrics_names = list(statistics.keys())

    if len(all_statistics) > 0 :
        all_statistics_dic = {}
        #
        for key_1 in ['train', 'test']:
            all_statistics_dic[key_1] = {}
            for key in statistics[key_1].keys():
                all_statistics_dic[key_1][key] = [statistics[key_1][key] for statistics in all_statistics ]
        #
        for key in statistics.keys():
            if key in ['train', 'test'] : continue
            all_statistics_dic[key] = [statistics[key] for statistics in all_statistics ]
    else :
        all_statistics_dic = {}
    
    return all_models_per_trials, all_statistics_dic