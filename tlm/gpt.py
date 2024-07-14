"""Module containing a simple implementation of the GPT model."""

from functools import reduce

from flax import linen

from .attention import AttentionLayer
from .types import Batched, Boolean, Vector

__all__ = ["TinyGPT"]


class TinyGPT(linen.Module):
    """Module containing a simple and tiny GPT model."""

    num_blocks: int
    vocabulary_size: int

    def setup(self) -> None:
        """Setup the ``TinyGPT`` object."""
        self._token_embedding = linen.Dense(features=256)
        self._blocks: list[DecoderBlock] = [
            DecoderBlock(out_features=256) for _ in range(self.num_blocks)
        ]
        self._output_layer = linen.Dense(features=self.vocabulary_size)

    def __call__(self, inputs: Batched[Vector], padding_mask: Boolean[Vector]) -> Batched[Vector]:
        """Return the forward pass of the ``TinyGPT`` model on some input data.

        Warning:
            This function returns the ``logits`` of the probability distribution. Make sure to
            compute ``softmax`` on the results to generate the probability distribution over tokens
            in the alphabet.

        Args:
            inputs (Batched[Vector]): Array of input vectors.
            padding_mask (Boolean[Vector]): Array containing the padding vector mask.

        Returns:
            Batched[Vector]: Logits of the probability distribution over vocabulary tokens.
        """
        embeddings = self._token_embedding(inputs)
        activations = reduce(
            lambda activations, layer: layer(activations, padding_mask), self._blocks, embeddings
        )
        return self._output_layer(activations)


class DecoderBlock(linen.Module):
    """Module containing the implementation of the decoder block in the GPT architecture.

    Note:
        A ``DecoderBlock`` contains a self-attention layer, followed by a feedforward layer.
        Residual connections are applied after each transformation and before the layer norms.
    """

    out_features: int
    num_heads: int = 4

    def setup(self) -> None:
        """Setup a ``DecoderBlock`` object."""
        self._attention = AttentionLayer(num_heads=self.num_heads, out_features=self.out_features)
        self._attention_norm = linen.LayerNorm()
        self._feedforward = linen.Dense(self.out_features)
        self._feedforward_norm = linen.LayerNorm()

    @linen.compact
    def __call__(self, inputs: Batched[Vector], padding_mask: Boolean[Vector]) -> Batched[Vector]:
        """Compute the forward pass of the ``DecoderBlock`` on some input data.

        Args:
            inputs (Batched[Vector]): Array of input vectors to process. Must have
                ``self.out_features`` features.
            padding_mask (Boolean[Vector]): Vector defining the mask eliminating masking on padded
                tokens.

        Returns:
            Batched[Vector]: Output of the ``DecoderBlock`` layer.
        """
        attention_acts = self._attention(inputs, padding_mask)
        attention_acts = _residual_connection(attention_acts, inputs)
        attention_acts = self._attention_norm(attention_acts)

        feedforward_acts = self._feedforward(attention_acts)
        feedforward_acts = _residual_connection(feedforward_acts, attention_acts)
        feedforward_acts = self._feedforward_norm(feedforward_acts)
        return feedforward_acts


def _residual_connection(
    activations: Batched[Vector], residuals: Batched[Vector]
) -> Batched[Vector]:
    """Return a new batch of vectors by applying some residuals to the activations.

    Note:
        ``activations`` and ``residuals`` must have the same dimensions.

    Args:
        activations (Batched[Vector]): Collection of activations.
        residuals (Batched[Vector]): Collection of residuals to apply in the connection.

    Returns:
        Batched[Vector]: Result of applying the residual connection between ``activations`` and
        ``residuals``.
    """
    return activations + residuals
