import os
import torch
from torch import nn
from time import time
from math import ceil

from typing import Callable
from typing import Any
from typing import Type
from typing import Union

from satforecast.modeling.model_selection import rolling_batch
from satforecast.data.data import MODEL_DIR

def tsize(t) -> float:
    """ Get size of tensor in GB """
    return t.element_size() * t.nelement() / (1024)**3

def print_sizes(**args):
    """ Print size of tensors provided as tesnor=tensor """
    print(
        ', '.join([f'Size of {k} = {tsize(v):.3f}GB ' for k, v in args.items()])
    )

def train(
    model: Type[nn.Module],
    model_name: str,
    criterion: Type[nn.Module],
    optimizer: Type[torch.optim.Optimizer],
    files_list: list[str],
    train_frac: float,
    val_frac: float,
    seq_len: int,
    batch_size: int,
    max_epochs: int,
    val_level: str = 'epoch',
    early_stopping: bool = True,
    min_improv: float = 0,
    max_iter_improv: int = 2,
    log_level: int = 2,
    predict: Callable[[Any], torch.Tensor] = None,
    predict_kwargs: dict[str, Any] = {},
    ) -> Union[tuple[list[float], list[float]], list[float]]:
    """
    Train and save a local image-to-image model
        - May use hugging face or skorch in the future

    Parameters
    ----------
    model: pytorch model
    criterion: loss function
    optimizer: optimization algorithm
    model_name: model saved as
                os.path.join(MODEL_DIR, model_name, '.', minor_version, '.pth')
    files_list: files containing image data (.npy)
    train_frac: fraction of files to use for training
    val_frac: fraction of files to use for validation
            - note train_frac + val_frac need not equal 1
    seq_len: number of previous images used to predict the next
    batch_size: number of sequences trained on simultaneously
    max_epochs: number of times to run over training set
    val_level: whether to validate at 'epoch' or 'batch' level
    early_stopping: whether to use early stopping
    min_improv: amount by which loss must decrease, else early stopping
    max_iter_improv: number of  allowed before requiring min_improv
    log_level: verbosity - how much to print

    Returns
    -------
        (train_losses, val_losses) if val_frac != 0
        train_losses if val_frac == 0
    """

    # Constants
    n_images = len(files_list)
    train_n = int(train_frac * n_images)
    images_per_batch = seq_len + batch_size
    n_batches = train_n / images_per_batch

    # Try writing first so we don't get an error after doing all the training
    minor_version = 0
    model_path = os.path.join(MODEL_DIR, f'{model_name}.{minor_version}.pth')
    os.makedirs(MODEL_DIR, exist_ok=True)
    while os.path.exists(model_path):
        minor_version += 1
        model_path = os.path.join(MODEL_DIR, f'{model_name}.{minor_version}.pth')

    torch.save(torch.tensor([1,2]), model_path) # No error handling for now
    if log_level > 1:
        print(f'Model will be saved to {model_path}')

    # Reused validation sets
    if val_frac != 0:
        time_start = time()
        X_val, y_val = rolling_batch(
                        files = files_list,
                        start = train_n,
                        stop = int((train_frac + val_frac)*n_images),
                        seq_len = seq_len
        )

        if log_level > 1:
            print(f'Time to make validation tensors: {time() - time_start}')
            print_sizes(X_val=X_val, y_val=y_val)
        val_losses = []

    # Begin training
    model.train()
    train_losses = []

    break_from_batch = False
    for epoch_n in range(max_epochs):

        epoch_start = time()
        for batch_n in range(int(n_batches)):

            # Enhancement TODO: Use smaller final batch if needed
            # Use range(ceil(n_batches))
            # if batch_n == ceil(n_batches):
            #    batch_size = train_n - int(n_batches) * images_per_batch - seq_len
            #    print(batch_size) # needs to be passes (and used) in rolling_batch

            # Make training batch
            start_batch_ind = batch_n * images_per_batch
            stop_batch_ind = (batch_n + 1) * images_per_batch
            X_train, y_train = rolling_batch(
                                                files = files_list,
                                                start = start_batch_ind,
                                                stop = stop_batch_ind,
                                                seq_len = seq_len
            )

            # Predictions and loss on training batch
            train_start = time()
            y_pred = model.predict(X_train)
            train_loss = criterion(y_pred, y_train)
            train_losses.append(train_loss.item())
            train_time = round(time() - train_start, 1)
            del X_train, y_train, y_pred

            # Update model
            update_start = time()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            update_time = round(time() - update_start, 1)
            if val_frac == 0 and log_level > 0:
                print(
                    f'Batch {batch_n}:',
                    f'Train loss = {train_losses[-1]},',
                    f'Train time = {train_time},',
                    f'Update time = {update_time}'
                )

            # Batch-level validation
            if val_level == 'batch' and val_frac != 0:
                val_start = time()
                model.eval()
                y_pred = model.predict(X_val)
                val_loss = criterion(y_pred, y_val)
                val_losses.append(val_loss.item())
                val_time = round(time() - val_start, 1)
                del y_pred, val_loss
                if  log_level > 0:
                    print(
                        f'Batch {batch_n}:',
                        f'Validation loss = {val_losses[-1]},',
                        f'Train time = {train_time},',
                        f'Update time = {update_time},',
                        f'Validation time = {val_time}'
                    )

                # Batch-level early stopping
                if early_stopping and len(val_losses) > max_iter_improv \
                        and val_losses[-(max_iter_improv + 1)] - min_improv \
                            <= min(val_losses[-max_iter_improv:]):
                    if log_level > 0:
                        print(
                            '\n!!! Early stopping triggered after',
                            f'{epoch_n} epochs and', # current not complete
                            f'{batch_n + 1} batches !!!\n'
                            )
                    break_from_batch = True
                    break

                model.train()

        # Don't continue to next epoch if batch-level early stopping triggered
        if break_from_batch:
            break

        epoch_time = round(time() - epoch_start, 1)

        # Epoch-level validation
        if val_level == 'epoch' and val_frac != 0:
            val_start = time()
            model.eval()
            y_pred = model.predict(X_val)
            val_loss = criterion(y_pred, y_val)
            val_losses.append(val_loss.item())
            val_time = round(time() - val_start, 1)
            del y_pred, val_loss
            if log_level > 0:
                print(
                    f'Epoch {epoch_n}:',
                    f'Validation loss = {val_losses[-1]},',
                    f'Epoch time = {epoch_time},',
                    f'Validation time = {val_time}'
                )

            # Epoch-level early stopping
            if early_stopping and len(val_losses) >= max_iter_improv \
                    and val_losses[-(max_iter_improv + 1)] - min_improv \
                        <= min(val_losses[-max_iter_improv:]):
                if log_level > 0:
                    print(
                        '\n!!! Early stopping triggered after',
                        f'{epoch_n + 1} epochs !!!\n'
                        )
                break

            model.train()

    # Save model
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict': criterion.state_dict()
        },
        model_path
        )

    if val_frac != 0:
        return train_losses, val_losses
    else:
        return train_losses