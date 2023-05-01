import numpy as np
import torch

from satforecast.modeling.model_selection import rolling_batch


class TestRollingBatch():

    def test_seq_len_2_batch_size_2(self):
        in_arr = np.arange(16).reshape(4,2,2)
        X_expected = torch.tensor([ # batch
                                    [ # seq
                                        [ # channels
                                            [[ 0,  1], # image
                                            [ 2,  3]]
                                        ],
                                        [
                                            [[ 4,  5],
                                            [ 6,  7]]
                                        ]
                                    ],
                                    [
                                        [
                                            [[ 4,  5],
                                            [ 6,  7]]
                                        ],
                                        [
                                            [[ 8,  9],
                                            [10, 11]]
                                        ]
                                    ]
                                ])
        y_expected = torch.tensor([ # batch
                                    [ # channels
                                        [[8,9], # label
                                        [10,11]],
                                    ],
                                    [
                                        [[12,13],
                                        [14,15]]
                                    ]
                                 ])

        X, y = rolling_batch(
                                images_arr=in_arr,
                                start=0,
                                stop=len(in_arr),
                                seq_len=2
                                )

        template = 'expected {} = {}\ngot\n{}'
        assert torch.equal(X, X_expected), template.format('X', X_expected, X)
        assert torch.equal(y, y_expected), template.format('y', y_expected, y)
