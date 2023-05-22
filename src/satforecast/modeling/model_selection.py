from itertools import product
from re import sub
from numpy import mean
from os import path
from json import dump
from torch import load

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
            name_base: str,
            files_list: list[str],
            param_grid: Union[dict[str, Any], list[dict[str, Any]]],
            criterion: Type[Module],
            criterion_params: dict[str, Any] = {},
            nests_per_config: int = 1,
            log_level = 3
        ) -> None:
        """
        Create grid of model and training configurations

        Parameters
        ----------
        model_class: class to test with various parameters (~sklearn estimator)
        name_base: a base for saved file names
            e.g. crnn_v0 -> crnn_v0__config_n.0.pth, crnn_v0__results.json
        files_list: files for training an validation - passed to train.train()
        param_grid: parameters over which to search for the best model/training
            For optimizer and scheduler classes, use 'optimizer' and
            'scheduler' keys respectively.
            For parameters, use '<model/optimizer/scheduler/etc.>__<param>'
            format for keys
            For models with normalization, use 'model__do_norm' and instead of
            providing a file list. The training files_list and train_frac
            within the nesting loop will assign the model train_list
        criterion: loss/scoring function class - analogous to sklearn scoring
        criterion_params: parameters to initialize criterion
        nests_per_config: number of nests to do NCV training
        log_level: verbosity - how much to print
        """
        self.model_class = model_class
        self.name_base = name_base
        self.files_list = files_list
        self.criterion = criterion
        self.criterion_params = criterion_params
        self.nests_per_config = nests_per_config
        self.log_level = log_level

        # Put dictionary of lists into list of dictionaries form if needed
        if type(param_grid) == dict:
            args = param_grid.keys()
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
                        k.split(delim)[1]: v for k, v in param_dict.items()
                        if (
                            k.split(delim)[0] == grid_name.split('_')[0]
                            and len(k.split(delim)) == 2
                            # Optimizer and scheduler classes pass first
                        )
                    }
                )

    def fit(self) -> None:
        """
        Train model with model and training configurations created in __init__

        Creates dictionary (self.results) containing lists of model_names,
        parameters, and training and validation scores
        """

        if self.log_level > 0:
            print(f'Performing NCV on {len(self.param_grid)} configurations')

        self.results = {}

        for (
                config_n,
                (
                    model_param, # _ indicates class, not instance
                    optimizer_, optimizer_param,
                    scheduler_, scheduler_param,
                    train_param
                )
            ) in enumerate(
                zip(
                    self.model_params,
                    self.optimizers, self.optimizer_params,
                    self.schedulers, self.scheduler_params,
                    self.train_params
                )
            ):

            # One name per config - only last nest model is saved
            config_name = self.name_base + f'__config_{config_n}'
            self.results[config_name] = {}
            config_dict = {
                'model_params': model_param,
                'criterion': self.criterion,
                'criterion_params': self.criterion_params,
                'optimizer': optimizer_,
                'optimizer_params': optimizer_param,
                'scheduler': scheduler_,
                'scheduler_params': scheduler_param,
                'train_params': train_param,
                'files_n': len(self.files_list)
            }
            # Types not json serializable so change to repr in config_dict
            for class_ in ('criterion', 'optimizer', 'scheduler'):
                config_dict[class_] = repr(config_dict[class_])

            if self.log_level > 0:
                print(f'Working on config_{config_n}')
                if self.log_level > 1:
                    from pprint import pprint
                    pprint(config_dict)

            # If model has normalizaion, remove 'do_norm', will set train_files
            do_norm_in_model = 'do_norm' in model_param
            if do_norm_in_model:
                do_norm = model_param['do_norm']
                del model_param['do_norm']

            # Set large loss and skip training if combining
            # do_norm = False and do_rev_norm = True
            if 'do_rev_norm' in model_param and model_param['do_rev_norm']:
                if not do_norm:
                    self.results[config_name][f'nest_{nest_n}'] = {
                        'train_loss': [999],
                        'val_loss': [999]
                    }
                    continue

            original_config_train_frac = train_param['train_frac']

            for nest_n in range(1, self.nests_per_config + 1):

                if self.log_level > 2:
                    print(f'nest {nest_n}')

                train_param['train_frac'] = (
                    nest_n / self.nests_per_config * original_config_train_frac
                )

                # Determine training files (train_n) as train.train()
                if do_norm_in_model and do_norm:
                    self.model_param['train_files'] = self.files_list[:
                        int(
                            train_param['train_frac']
                            * len(self.files_list)
                        )
                    ]

                # Initialize main objects
                model = self.model_class(**model_param)
                criterion = self.criterion(**self.criterion_params)
                optimizer = optimizer_(model.parameters(), **optimizer_param)
                if scheduler_ is not None:
                    scheduler = scheduler_(optimizer, **scheduler_param)
                else:
                    scheduler = None

                # Train
                model_path, train_loss, val_loss = train(
                    model,
                    config_name,
                    criterion,
                    optimizer,
                    files_list = self.files_list,
                    **train_param,
                    scheduler = scheduler,
                    save_model = (nest_n == self.nests_per_config)
                )

                # Nest results
                self.results[config_name][f'nest_{nest_n}'] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }

            # Config results
            self.results[config_name]['mean_final_val_loss'] = mean([
                nest['val_loss'][-1]
                for nest in self.results[config_name].values()
            ])

            if self.log_level > 0:
                print('Mean final validation loss:',
                    self.results[config_name]['mean_final_val_loss']
                )

            # Save config and config results to accompanying json
            config_file = path.splitext(model_path)[0] + '.json'
            config_dict['results'] = self.results[config_name]
            with open(config_file, 'w') as cf:
                dump(config_dict, cf)

        self._make_best() # Split into own method; call automatically

    def _make_best(self):
        """ Determine the best model and set relevant attributes """

        self.best_config = min(
            self.results,
            key = lambda config: self.results[config]['mean_final_val_loss']
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

        model_path = path.join(
            MODEL_DIR, f'{self.best_config}.{minor_version}.pth'
            )
        checkpoint = load(model_path)
        model = self.model_class(**self.best_model_params)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model
