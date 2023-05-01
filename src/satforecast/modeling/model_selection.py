import numpy as np
import torch

def rolling_batch():

    # Load images
    batch_images = np.array([plt.imread(files_list[i]) for i in range(start, stop)])
    image_size = batch_images.shape[-2:]

    # Explicit single gs channel
    batch_images = batch_images.reshape(len(batch_images), 1, *image_size)

    # Organize into (batch_size, seq_len, channels=1, *image_size) (X and y)
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
