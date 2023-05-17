from itertools import product

from typing import Any
from typing import Type
from typing import Union
from torch.nn import Module

from satforecast.modeling.train import train

class GridSearch():
    """ Search over model and training parameters """
    def __init__(
        model_class: Type[Module],
        model_name_base: str,
        param_grid: Union[dict[str, Any], list[dict[str, Any]]]:
    ) -> None:
    """
    Create grid of model and training configurations

    Parameters
    ----------
    model_class: class to test with various parameters (like sklearn estimator)
    model_name_base: a base to which parameter information is added
        e.g. crnn -> crnn__seq_len-5.pth
    param_grid: parameters over which to search for the best model/training
        use '<model/criterion/etc.>__<param>' format for the keys
    """
    self.model_class = model_class
    self.model_name_base = model_name_base

    # Put dictionary of lists into list of dictionaries form if needed
    if type(param_grid) == dict:
        args = parm_grid.keys()
        value_combinations = product(*param_grid.values())
        self.param_grid = [
            {arg: value for arg, value in zip(args, values)}
            for values in value_combinations
        ]
    elif type(param_grid) == list:
        self.param_grid = param_grid

    # Split param_grid into smaller grids to feed to independent constructors


    def fit(self) -> None:
    """
    Train a model with model and training configurations created in __init__

    Creates dictionary (self.results) containing lists of model_names,
    parameters, and training and validation scores
    """
    for params in self.param_grid:


    def predict(self, x):

