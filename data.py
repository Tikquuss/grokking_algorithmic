import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split

def get_arithmetic_data(
    p: int,
    q: int,
    operator: str,
    r_train: float,
    batch_size, 
    eval_batch_size,
    seed: int = None
):
    """
    Generates a dataset of arithmetic expressions with variable operand counts.
    Each sample is ( (a, b), c) with a, b in [0, p) and c = (a operator b) % q in [0, q).

    Parameters:
        p (int): The modulo for arithmetic operations.
        q (int): The modulo for the results
        operator (str): The operation to use ("+", "-", "*").
        r_train (float): Fraction of data to use for training (between 0 and 1).
        batch_size (int or None): Batch size for training. If None, uses the entire training set.
        eval_batch_size (int or None): Batch size for evaluation. If None, uses the entire evaluation set.
        seed (int): Random seed for shuffling.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders.
         Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Raw train/test splits (X_train, X_test, y_train, y_test).
    """
    assert p > 0 and q > 0, "p and q must be positive integers."
    assert operator in ["+", "-", "*"], "Invalid operator. Must be one of '+', '-', '*'"
    assert 0 < r_train <= 1.0, "r_train must be in the range (0, 1]."

    data = list(itertools.product(range(p), range(p)))
    X, y = [], []
    for x1, x2 in data:
        #x = torch.cat([F.one_hot(torch.tensor(x1), num_classes=p), F.one_hot(torch.tensor(x2), num_classes=p)]).float() # (2*p,)
        x = torch.stack([F.one_hot(torch.tensor(x1), num_classes=p), F.one_hot(torch.tensor(x2), num_classes=p)]).float() # (2, p)
        X.append(x)
        y.append(eval(f"({x1} {operator} {x2}) % {p}"))
    X, y = torch.stack(X), torch.tensor(y) # (p^2, 2, p), (p^2,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=r_train, random_state=seed, stratify=None)

    N_train, N_test = len(X_train), len(X_test)
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    batch_size = N_train if batch_size is None else min(batch_size, N_train)
    eval_batch_size_train = N_train if eval_batch_size is None else min(eval_batch_size, N_train)
    eval_batch_size_test = N_test if eval_batch_size is None else min(eval_batch_size, N_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    train_loader_for_eval = DataLoader(train_set, batch_size=eval_batch_size_train, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size_test, shuffle=False, drop_last=False, num_workers=0)

    
    print(f"Data size : train = {N_train}, test = {N_test}")
    print(f"Loader size : train = {len(train_loader)}, train for val = {len(train_loader_for_eval)}, test = {len(test_loader)}")

    return (train_loader, train_loader_for_eval, test_loader), (X_train, X_test, y_train, y_test)