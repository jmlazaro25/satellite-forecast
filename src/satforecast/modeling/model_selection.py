from itertools import product
from re import sub
from numpy import mean
from os.path import join

from typing import Any
from typing import Type
from typing import Union
from torch.nn import Module

from satforecast.modeling.train import train
from satforecast.data.data import MODEL_DIR

class GridSearchNCV():
    """ Search over model and training parameters """
    def __init__(
            self,
            model_class: Type[Module],
            model_name_base: str,
            param_grid: Union[dict[str, Any], list[dict[str, Any]]],
            criterion: Type[Module],
            criterion_params: dict[str, Any] = {},
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
        """
        self.model_class = model_class
        self.model_name_base = model_name_base
        self.criterion = criterion
        self.criterion_params = criterion_params

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
        self.grid_names = (
            'model_params',
            'optimizer_params',
            'scheduler_params',
            'train_params'
            )
        for grid_name in self.grid_names:
            setattr(self, grid_name, [])
        for param_dict in self.param_grid:
            self.optimizers.append(param_dict[optim_str])
            if sched_str in param_dict:
                self.schedulers.append(param_dict[sched_str])
            else:
                self.schedulers.append(None)
            for grid_name in self.grid_names:
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

        self.results = {}

        # Naming helper functions
        def params_str(params):
            out = ''
            for k, v in params.items():
                if k == 'files_list':
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
                optimizer_, optimizer_param, # _ indicates class not instance
                scheduler_, scheduler_param,
                train_param
            ) in zip(
                self.model_params,
                self.optimizers, self.optimizer_params,
                self.schedulers, self.scheduler_params,
                self.train_params
        ):

            # One name per config - only last nest in saved (for now)
            config_name = (self.model_name_base
                + params_str(model_param)
                + obj_str('optim', optimizer_) + params_str(optimizer_param)
                + obj_str('sched', scheduler_) + params_str(scheduler_param)
                + params_str(train_param)
                )

            self.results[config_name] = {}

            for nest_n in range(1, nests_per_config + 1):

                train_param['train_frac'] *= nest_n / nests_per_config

                model = self.model_class(**model_param)
                criterion = self.criterion(**self.criterion_params)
                optimizer = optimizer_(model.parameters(), **optimizer_param)
                if scheduler_ is not None:
                    scheduler = scheduler_(optimizer, **scheduler_param)

                train_loss, val_loss = train(
                    model, config_name, criterion, optimizer, **train_param
                    save_model = (nest_n == nests_per_config)
                )

                self.results[config_name][f'nest_{nest_n}'] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }

            self.results[config_name]['mean_final_val_loss'] = mean([
                nest['val_loss'][-1] for nest in self.results[config_name]
            ])

        self._make_best() # Split into own method; call automatically

    def _make_best(self):
        """ Determine the best model and set relevant attributes """

        self.best_config = max(
            self.results,
            key = lambda config: config['mean_final_val_loss']
        )
        self.best_loss = self.results[self.best_config]['mean_final_val_loss']

        self.best_index = list(self.results).index(self.best_config) # (3.7+)
        for grid_name in self.grid_names:
            setattr(
                self,
                f'best_{grid_name}',
                getattr(self, grid_name)[self.best_index]
            )

    def load_best(self, minor_version: int = 0):
        """ Load and return the best model """

        model_path = join(MODEL_DIR, f'{self.best_config}.{minor_version}.pth')
        checkpoint = torch.load(model_path)
        model = self.model_class(**self.best_model_params)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model
