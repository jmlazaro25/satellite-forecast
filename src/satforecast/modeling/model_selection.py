import numpy as np
import torch
from typing import Iterable

def rolling_batch(
    files: Iterable[str] = None,
    images_arr: np.ndarray = None,
    start: int = None,
    stop: int = None,
    seq_len: int = None
    ):
    """
    Make feature and label tensors.
    If both files and images_arr are provided, images_arr is used.
    Might turn into class when dealing with multiple modes/channels

    Parameters
    ----------
    files: file paths to load
    images_arr: numpy array of shape (n_images, *image_size) or
                (n_images, n_channels, *image_size) if more than 1 channel
    start: index in files or images_arr to start making batch
    stop: index + 1 in files or images_arr to use as the last label
    seq_len: number of images used to predict next image (label)

    Returns
    -------
    torch.tensor of shape (batch_size, seq_len, channels, *image_size)
    torch.tensor of shape (batch_size, channels, *image_size)
    """

    # Determine images used for batch
    if images_arr is not None:
        batch_images = images_arr
    elif files is not None:
        batch_images = np.array([np.load(files[i]) for i in range(start, stop)])

    # Explicit single gs channel
    image_size = batch_images.shape[-2:]
    batch_images = batch_images.reshape(len(batch_images), 1, *image_size)

    # Organize X into (batch_size, seq_len, channels=1, *image_size)
    # And y into (batch_size, channels=1, *image_size)
    batch_size = stop - start - seq_len
    X = np.array([
                    batch_images[seq_n : seq_n + seq_len]
                    for seq_n in range(batch_size)
                    ])
    y = np.array([
                    batch_images[seq_n + seq_len]
                    for seq_n in range(batch_size)
                    ])

    return torch.from_numpy(X), torch.from_numpy(y)

def batch_seq_in_channels(
    pass
