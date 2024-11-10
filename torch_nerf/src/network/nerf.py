"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.input_layer = nn.Sequential(
            nn.Linear(pos_dim, feat_dim),
            nn.ReLU()
        )

        # 4 hidden layers
        self.hidden_layers1 = nn.Sequential(*[
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU())
            for _ in range(4)
        ])

        # skip layer
        self.skip_layer = nn.Sequential(
            nn.Linear(pos_dim + feat_dim, feat_dim),
            nn.ReLU()
        )

        #hidden layers
        self.hidden_layers2 = nn.Sequential(*[
            nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU())
            for _ in range(4)
        ])

        # density layer (feat_dim -> 1)
        self.density_layer = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.ReLU()
        )

        # radiance layer
        self.rgb_layer = nn.Sequential(
            nn.Linear(view_dir_dim + feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 3),
            nn.Sigmoid()
        )

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        x = self.input_layer(pos)
        x = self.hidden_layers1(x)

        x = torch.cat([x, pos], dim=-1)
        x = self.skip_layer(x)
        x = self.hidden_layers2(x)

        sigma = self.density_layer(x)

        view_input = torch.cat([x, view_dir], dim=-1)
        radiance = self.rgb_layer(view_input)

        return sigma, radiance
