"""Module containing a simple implementation of the GPT model."""

from functools import reduce

from flax import linen

from .attention import AttentionLayer
from .encode import VOCABULARY_SIZE
from .types import Batched, Vector

__all__ = ["FeedForward"]


class TinyGPT(linen.Module):
    """Module containing a simple and tiny GPT model."""

    num_blocks: int

    def setup(self) -> None:
        """Setup the ``TinyGPT`` object."""
        self._token_embedding = linen.Dense(features=256)
        self._blocks: list[DecoderBlock] = [
            DecoderBlock(out_features=256) for _ in range(self.num_blocks)
        ] + [linen.Dense(features=VOCABULARY_SIZE)]

    def __call__(self, inputs: Batched[Vector]) -> Batched[Vector]:
        """Return the forward pass of the ``TinyGPT`` model on some input data."""
        embeddings = self._token_embedding(inputs)
        return reduce(lambda activations, layer: layer(activations), self._blocks, embeddings)


class DecoderBlock(linen.Module):
    """Module containing the implementation of the decoder block in the GPT architecture."""

    out_features: int
    num_heads: int = 4

    def setup(self) -> None:
        """Setup a ``DecoderBlock`` object."""
        self._attention = AttentionLayer(num_heads=self.num_heads, out_features=self.out_features)
        self._feedforward = linen.Dense(self.out_features)

    @linen.compact
    def __call__(self, inputs: Batched[Vector]) -> Batched[Vector]:
        """Compute the forward pass of the ``DecoderBlock`` on some input data."""
        activations_1 = self._attention(inputs, masked=True)
        activations_2 = linen.LayerNorm()(activations_1 + inputs)
        activations_3 = self._feedforward(activations_2)
        activations_4 = linen.LayerNorm()(activations_3 + activations_2)
        return activations_4


class FeedForward(linen.Module):
    """Class containing the implementation of the standard residual feed forward."""

    out_features: int
    num_layers: int

    def setup(self) -> None:
        """Setup the ``FeedForward`` object."""
        self._perceptrons = [linen.Dense(self.out_features) for _ in range(self.num_layers)]

    @linen.compact
    def __call__(self, inputs: Batched[Vector]) -> Batched[Vector]:
        """Return the result of the forward pass of the ``FeedForward`` module."""
        activations = inputs
        residuals: Batched[Vector] | None = None
        for _ in range(self.num_layers):
            activations = linen.Dense(self.out_features)(activations)
            if residuals is not None:
                activations = activations + residuals
            activations = linen.LayerNorm()(activations)
            activations = linen.relu(activations)
            residuals = activations
        return activations
