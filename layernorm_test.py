import torch
import torch.nn.functional as F

a = torch.tensor(
    [
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
            ],
            [
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
            ],
            [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
        ],
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
            ],
            [
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
            ],
            [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
        ],
    ],
    dtype=torch.float32,
)
print(a.shape)
a_lp_norm = F.normalize(a, dim=1)
print(a_lp_norm)

# LayerNorm
layer_norm = torch.nn.LayerNorm(a.shape[1:])
a_layer_norm = layer_norm(a)
print(a_layer_norm)

# BatchNorm
batch_norm = torch.nn.BatchNorm2d(a.shape[1])
a_batch_norm = batch_norm(a)
print(a_batch_norm)
