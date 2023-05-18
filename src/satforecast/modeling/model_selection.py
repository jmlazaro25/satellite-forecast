from itertools import product
from os.path import split
from re import sub

from typing import Any
from typing import Type
from typing import Union
from torch.nn import Module

from satforecast.modeling.train import train

class GridSearchCV():
    """ Search over model and training parameters """
    def __init__(
            model_class: Type[Module],
            model_name_base: str,
            param_grid: Union[dict[str, Any], list[dict[str, Any]]],
            criterion: Type[Module],
            criterion_params: dict[str, Any] = {},
            lower_is_better: bool = True,
            nests_per_config: int = 1
        ) -> None:
        """
        Create grid of model and training configurations

        Parameters
        ----------
        model_class: class to test with various parameters (~sklearn estimator)
        model_name_base: a base to which parameter information is added
            e.g. crnn -> crnn__seq_len-5.pth
        param_grid: parameters over which to search for the best model/training
            For optimizer and scheduler classes, use 'optimizer' and
            'scheduler' keys respectively. For parameters, use
            '<model/criterion/optimizer/etc.>__<param>' format for keys
        criterion: loss/scoring function class - analogous to sklearn scoring
        criterion_params: parameters to initialize criterion
        lower_is_better: whether a lower criterion value is better
        """
        self.model_class = model_class
        self.model_name_base = model_name_base
        self.criterion = criterion
        self.criterion_params = criterion_params
        self.lower_is_better = lower_is_better

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

        # Split param_grid into smaller grids for independent constructors
        delim = '__'
        optim_str = 'optimizer'
        sched_str = 'scheduler'
        self.optimizers = []
        self.schedulers = []
        grid_names = (
            'model_params',
            'optimizer_params',
            'scheduler_params',
            'train_params'
            )
        for grid_name in grid_names:
            setattr(self, grid_name) = []
        for param_dict in self.param_grid:
            self.optimizers.append(param_dict[optim_str])
            if sched_str in param_dict:
                self.schedulers.append(param_dict[sched_str])
            else:
                self.schedulers.append(None)
            for grid_name in grid_names:
                getattr(self, grid_name).append(
                    {
                        k.split(delim)[1]: value for k, v in param_dict.items()
                        if k.split(delim)[0] == grid_name
                    }
                )

    def fit(self) -> None:
        """
        Train model with model and training configurations created in __init__

        Creates dictionary (self.results) containing lists of model_names,
        parameters, and training and validation scores
        """

        # Naming helper functions
        def params_str(params):
            out = ''
            for k, v in params.items():
                if k == 'files_list'
                    k = 'n_files' # Save len(files_list) rather than whole list
                    v = len(files_list)
                if k != 'log_level':
                    continue
                out += f'__{k}-{v}'
            return out

        def obj_str(cat_name, obj):
            return f'__{cat_name}' + sub(r'[()]', '', repr(obj))

        for (
                model_param,
                optimizer, optimizer_param,
                scheduler, scheduler_param,
                train_param
            ) in zip(
                self.model_params,
                self.optimizers, self.optimizer_params,
                self.schedulers, self.scheduler_params,
                self.train_params
        ):
            model = self.model_class(**model_param)
            criterion = self.criterion(**self.criterion_params)
            optimizer = optimizer(model.parameters(), **optimizer_param)
            if scheduler is not None:
                scheduler = scheduler(optimizer, **scheduler_param)

            model_name = (self.model_name_base
                + params_str(model_param)
                + obj_str('optim', optimizer) + params_str(optimizer_param)
                + obj_str('sched', scheduler) + params_str(scheduler_param)
                + params_str(train_param)
            )

            train(model, model_name, criterion, optimizer, **train_param)

    def predict(self, x):

