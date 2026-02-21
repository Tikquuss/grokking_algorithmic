import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import numpy as np

import random

import copy
from tqdm import tqdm
import os
from collections import defaultdict
import inspect
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable, Iterable

########################################################################################
########################################################################################

def get_loss(model, batch_x, batch_y, criterion) :
    model.train()
    scores = model(batch_x)
    
    batch_y = batch_y.squeeze()
    scores = scores.squeeze()

    # # Reshape batch_y to (batch_size, 1) if it's (batch_size,)
    # if batch_y.dim() == 1:
    #     batch_y = batch_y.unsqueeze(1) # (batch_size, _)
    # # Ensure scores also have the same shape as batch_y
    # if scores.shape != batch_y.shape:
    #     scores = scores.view_as(batch_y) # (batch_size, _)

    loss = criterion(scores, batch_y)
    return loss, scores

########################################################################################
########################################################################################

@torch.no_grad()
def eval_model_classification(model, loader, criterion, device):
    model.eval()

    acc = 0
    loss = 0
    n = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        scores = model(batch_x).squeeze()
        n += batch_x.shape[0]
        loss += criterion(scores, batch_y).item() * batch_x.shape[0]
        _, preds = scores.max(1)
        acc += (preds == batch_y).sum().item()
    return {"loss" : loss / n, "accuracy": acc / n}#, "n":n}

########################################################################################
########################################################################################

@torch.no_grad()
def eval_model_regression(model, loader, criterion, device):
    model.eval()
    
    loss = 0
    n = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        scores = model(batch_x)

        # Reshape batch_y to (batch_size, 1) if it's (batch_size,)
        if batch_y.dim() == 1:
            batch_y = batch_y.unsqueeze(1) # (batch_size, _)
        # Ensure scores also have the same shape as batch_y
        if scores.shape != batch_y.shape:
            scores = scores.view_as(batch_y) # (batch_size, _)

        n += batch_x.shape[0]
        loss += criterion(scores, batch_y).item() * batch_x.shape[0]

    return {"loss" : loss / n, "accuracy": 0}

########################################################################################
########################################################################################

def train(
    model, 
    train_loader:Iterable, 
    train_loader_for_eval:Iterable, 
    test_loader:Iterable, 
    criterion:Callable, 
    optimizer:None,
    device:None, 
    n_steps:int,
    get_loss:Callable, 
    eval_model:Callable,
    file_name:str, 
    checkpoint_path:str=None,
    get_other_metrics:Callable=None,
    eval_first:int=0,
    eval_period:int=1,
    print_step:int=1,
    save_model_step:int=1,  
    save_statistic_step:int=1,  
    state_path:str="",
    n_decimal = 6,
    verbose=True,
    ):
    """
    model (nn.Module) : The model to train
    train_loader (DataLoader) : Training data loader
    train_loader_for_eval (DataLoader) : Training data loader (for evaluation)
    test_loader (DataLoader) : Test/Val data loader
    criterion (nn.Module) : Loss function
    optimizer (Optimizer) : Optimizer
    device (str) : Device (cpu, cuda, cuda:0, etc)
    n_steps (int) : Number of training steps
    get_loss (Callable) : Callable to get the loss
    eval_model (Callable) : Callable to evaluate the model
    file_name (str) : file name for checkpointing (CNN, MLP, etc.)
    checkpoint_path (str) : Path to save the model checkpoints ("/path/to/experiment")
    get_other_metrics (Callable) : Callable for additional metrics, if needed
    eval_first (int) : Number of consecutive evaluation step at the beginning of training
    eval_period (int) : Evaluation frequency
    print_step (int) : Print frequency
    save_model_step (int) : Step interval to save model checkpoints
    save_statistic_step (int) : Step interval to save statistics (train/test loss, accuracy, etc.)
    state_path (str) : Path to load the model state before training if it exists
    verbose (bool) : Verbosity of the training
    """

    ##############
    # Checkpoint path
    if checkpoint_path is None : checkpoint_path=DIR_PATH_FIGURES
    os.makedirs(checkpoint_path, exist_ok=True)

    ##############
    # Number of training epochs
    n_batchs = len(train_loader)
    total_epochs = (n_steps + n_batchs - 1) // n_batchs
    n_steps = total_epochs * n_batchs
    
    if verbose :
        print(f"Number of training epochs & steps: {total_epochs} {n_steps}")

    ##############
    # Load the model state if it exists
    if os.path.exists(state_path):
        if verbose : print(f"Loading model state from {state_path}")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

    ##############

    all_metrics = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["train"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["test"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["steps_epoch"] = {}

    ##############

    train_statistics = eval_model(model, train_loader_for_eval, criterion, device)
    for k, v in train_statistics.items() :
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, criterion, device)
    for k, v in test_statistics.items() :
        all_metrics["test"][k].append(v)

    t_0 = 0
    all_metrics["all_steps"].append(t_0)
    all_metrics["steps_epoch"][t_0] = 0

    # ######################
    other_metrics={}
    # TODO
    if get_other_metrics is not None :
        for i, (batch_x, batch_y) in enumerate(train_loader) :
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss, batch_y_hat = get_loss(model, batch_x, batch_y, criterion)
            ##
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ###
            other_metrics = get_other_metrics(model, batch_x, batch_y, batch_y_hat, loss)
            for k, v in other_metrics.items() :
                all_metrics[k].append(v)
            break
    ######################
    
    STATE_NAME = 'model'
    STATE_NAME = 'state'
    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    acc = round(test_statistics['accuracy'], n_decimal)
    loss = round(test_statistics['loss'], n_decimal)
    torch.save(state, f"{checkpoint_path}/{file_name}_{STATE_NAME}_{t_0}_acc={acc}_loss={loss}.pth")
    #torch.save(state, f"{checkpoint_path}/{file_name}-step={t_0}-test_acc={acc}-test_loss={loss}-train_acc={train_statistics['accuracy']}-train_loss={train_statistics['loss']}.pth")
    
    ##############

    if verbose :
        to_print = "\n" + f"Step {0}/{n_steps}"
        to_print += " | " + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
        to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
        to_print += " | " + " | ".join(f'{k}: {v:.6f}' for k, v in other_metrics.items())
        print(to_print)

    ##############

    cur_step = 1 + t_0 
    tol_step = 0
    for epoch in tqdm(range(1, total_epochs+1), desc="Training (epochs)...", total=total_epochs):
        for i, (batch_x, batch_y) in enumerate(train_loader) :
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss, batch_y_hat = get_loss(model, batch_x, batch_y, criterion)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(closure=None)
        
            if cur_step in [1, n_steps] or cur_step % eval_period == 0 or cur_step <= eval_first:
                train_statistics = eval_model(model, train_loader_for_eval, criterion, device)
                for k, v in train_statistics.items() :
                    all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, criterion, device)
                for k, v in test_statistics.items() :
                    all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step+t_0)
                all_metrics["steps_epoch"][cur_step+t_0] = epoch

                if get_other_metrics is not None :
                    other_metrics = get_other_metrics(model, batch_x, batch_y, batch_y_hat, loss)
                    for k, v in other_metrics.items() :
                        all_metrics[k].append(v)
                
            if  verbose and (cur_step in [1, n_steps] or cur_step%print_step==0) :
                to_print = "\n" + f"Step {cur_step}/{n_steps}"
                to_print += " | " + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
                to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
                to_print += " | " + " | ".join(f'{k}: {v:.6f}' for k, v in other_metrics.items())
                print(to_print)

            if cur_step in [1, n_steps] or cur_step%save_model_step==0 or cur_step <= eval_first : 
                #state = model.state_dict()
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                acc = round(test_statistics['accuracy'], n_decimal)
                loss = round(test_statistics['loss'], n_decimal)
                torch.save(state, f"{checkpoint_path}/{file_name}_{STATE_NAME}_{cur_step+t_0}_acc={acc}_loss={loss}.pth")
                #torch.save(state, f"{checkpoint_path}/{file_name}-step={cur_step+t_0}-test_acc={test_statistics['accuracy']}-test_loss={test_statistics['loss']}-train_acc={train_statistics['accuracy']}-train_loss={train_statistics['loss']}.pth")

            if cur_step in [1, n_steps] or cur_step%save_statistic_step==0:
                #to_save = {k:v for k, v in all_metrics.items()}
                to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
                torch.save(to_save, f"{checkpoint_path}/{file_name}.pth")

            cur_step += 1


    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    acc = round(test_statistics['accuracy'], n_decimal)
    loss = round(test_statistics['loss'], n_decimal)
    torch.save(state, f"{checkpoint_path}/{file_name}_{STATE_NAME}_{cur_step+t_0}_acc={acc}_loss={loss}.pth")
    #torch.save(state, f"{checkpoint_path}/{file_name}-step={cur_step+t_0}-test_acc={test_statistics['accuracy']}-test_loss={test_statistics['loss']}-train_acc={train_statistics['accuracy']}-train_loss={train_statistics['loss']}.pth")

    for k, v in train_statistics.items() : all_metrics["train"][k].append(v)
    for k, v in test_statistics.items() : all_metrics["test"][k].append(v)
    all_metrics["all_steps"].append(cur_step+t_0)
    all_metrics["steps_epoch"][cur_step+t_0] = epoch
    if get_other_metrics is not None :
        for k, v in other_metrics.items() : all_metrics[k].append(v)

    #to_save = {k:v for k, v in all_metrics.items()}
    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{file_name}.pth")

    return all_metrics

########################################################################################
########################################################################################


def fix_experiment_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################################################################################
########################################################################################

def check_integer_and_return(s):
    """
    Checks if the input `s` is an integer or a string that can be cast to an integer.
    
    - If `s` is already an integer, it returns the integer.
    - If `s` is a string that can be cast to an integer, it returns the integer.
    - If `s` is a string but cannot be cast to an integer, it returns `None`.
    
    Parameters:
    s (int or str): The input value to check and possibly convert.
    
    Returns:
    int or None: The integer value if successful, otherwise `None`.
    """
    # Check if it's already an integer
    if isinstance(s, int):
        return s  # Return the integer as is
    
    # If it's a string, check if it can be cast to an integer
    elif isinstance(s, str):
        try:
            # Attempt to convert the string to an integer
            int_value = int(s)
            return int_value  # Return the integer
        except ValueError:
            # If casting to integer fails, return None
            return None

    # If it's neither an integer nor a string, return None
    return None

########################################################################################
########################################################################################

def run_experiments(args) :
    """
    args = {
        "model": nn.Module,  # Model to train
        "train_loader": DataLoader,  # Training data loader
        "train_loader_for_eval": DataLoader,  # Training data loader (for evaluation)
        "test_loader": DataLoader,  # Test/Val data loader
        "optimizer": torch.optim.Optimizer,  # Optimizer to use (e.g., Adam, SGD, etc.)
        "criterion": torch.nn.Module,  # Loss criterion (e.g., CrossEntropyLoss, MSELoss, etc.)
        "exp_dir": str,  # Experiment directory path ("/path/to/experiment")
        "file_name": str,  # file name for checkpointing (CNN, MLP, etc.)
        "exp_id": object,  # Experiment ID (e.g., 0, 1, etc.)
        "device": str ,  # device (cpu, cuda, cuda:0, etc)
        "get_exp_name_function": Callable,  # Function to get the experiment name
        "get_loss": Callable,  # Callable to get the loss
        "eval_model": Callable,  # Callable to evaluate the model
        "get_other_metrics": Callable(default=None),  # Callable for additional metrics, if needed
        "seed": int(default=0),  # Seed for reproducibility
        "n_steps": int,  # Number of training steps
        "eval_first": int(default=1),  # Number of consecutive evaluation step at the beginning of training
        "eval_period": int(default=1),  # Evaluation frequency
        "print_step": int(default=1),  # Print frequency
        "save_model_step": int(default=1),  # Step interval to save model checkpoints
        "save_statistic_step": int(default=1),  # Step interval to statistics (train/test loss, accuracy, etc.)
        "early_stopping_str": str(default=""),  # Early stopping configuration (if applicable)
        "state_path": str(default=""), # Path to load the model state before training if it exists
        "verbose": bool(default=True),  # Verbosity of the training
    }
    """

    exp_seed = args.get('seed', None)
    if exp_seed is not None :
        fix_experiment_seed(exp_seed)
    verbose = args.get('verbose', True)

    exp_id = args.get('exp_id', None)
    if exp_id is None :
        exp_id = 0
        args['exp_id'] = exp_id
        exp_name = args['get_exp_name_function'](args)

        ## If the experiment id was already use, create another one
        current_id = exp_id
        exp_id_int = check_integer_and_return(current_id)
        args['exp_id'] = current_id if exp_id_int is None else exp_id_int
        i=1
        while os.path.exists(os.path.join(args['exp_dir'], exp_name)):
            args['exp_id'] = f"{current_id}_{i}" if exp_id_int is None else (exp_id_int+i)
            exp_name = args['get_exp_name_function'](args)
            i+=1
    else :
        exp_name = args['get_exp_name_function'](args)

    args['checkpoint_path'] = os.path.join(args['exp_dir'], exp_name)
    os.makedirs(args['checkpoint_path'], exist_ok=True)

    ## Print parameters
    if verbose :
        print("=="*60)
        for k, v in args.items() :
            if k in ["model", "get_loss", "eval_model", "get_exp_name_function", "run_experiments", "get_other_metrics"] : continue
            if k in ["X_train", "y_train", "X_test", "y_test", "train_loader", "train_loader_for_eval", "test_loader"] : continue
            print(k, ":", v)
        print("=="*60)

    
    model = args['model'].to(args['device'])
    if verbose : 
        print("Model :", model, "\n")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of model trainable parameters : {n_params}")

    ## Train
    all_metrics = train(
        model, 
        args['train_loader'], args['train_loader_for_eval'], args['test_loader'], 
        criterion=args['criterion'], 
        optimizer=args['optimizer'],
        device=args['device'], 
        n_steps=args['n_steps'],
        get_loss=args['get_loss'], 
        eval_model=args['eval_model'],
        file_name=args['file_name'], 
        checkpoint_path=f"{args['exp_dir']}/{exp_name}",
        get_other_metrics=args.get('get_other_metrics', None),
        eval_first=args.get('eval_first', 1), 
        eval_period=args.get('eval_period', 1), 
        print_step=args.get('print_step', 1),
        save_model_step=args.get('save_model_step', 1),
        save_statistic_step=args.get('save_statistic_step', 1),
        state_path=args.get('state_path', ""),
        verbose=verbose,
    )

    return args, model, all_metrics

########################################################################################
########################################################################################

# Set the working directory to the parent directory of your top-level package
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from checkpointing import get_all_checkpoints_per_trials

def train_m_models(args, M:int=None, seeds:list=None):
    """Train M models and plot the loss and accuracies of each model separately."""
    assert M is not None or seeds is not None, "Either M or seeds should be provided."
    if seeds is not None:
        M = len(seeds)
    else :
        seeds = [args['seed'] + m if args['seed'] is not None else None for m in range(M)]
    all_checkpoint_paths = []
    for seed, m in zip(seeds, range(M)):
        print(f"Model {m+1}/{M}")
        args['exp_id'] = m # Set the experiment id
        args['seed'] = seed # Set the seed
        ## TODO : model, optimizer, criterion, train_loader, train_loader_for_eval, test_loader
        args_, model, all_metrics = run_experiments(args)
        all_checkpoint_paths.append(args_['checkpoint_path'])
        
    all_models_per_trials, all_metrics = get_all_checkpoints_per_trials(
        all_checkpoint_paths=all_checkpoint_paths, 
        exp_name=args['file_name'], 
        just_files=True, 
        verbose=args['verbose']
    )

    args['all_checkpoint_paths'] = all_checkpoint_paths
    return args, all_models_per_trials, all_metrics