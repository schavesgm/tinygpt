"""Module containing functionality to implement attention layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen

from .types import Batched, Boolean, Matrix, Vector

__all__ = [
    "create_causal_attention_mask",
    "create_padding_attention_mask",
    "compute_attention",
    "AttentionLayer",
]

# Constant to ensure the mask is correctly applied by ground masked values towards 0.0 probability
ENSURE_MASKED_IS_ZERO: float = -1e9


class AttentionLayer(linen.Module):
    """Class defining an attention layer module."""

    num_heads: int
    inner_features: int = 128
    out_features: int = 128

    def setup(self) -> None:
        """Setup the ``AttentionLayer`` object."""
        self._k_layer = linen.Dense(self.num_heads * self.inner_features)
        self._q_layer = linen.Dense(self.num_heads * self.inner_features)
        self._v_layer = linen.Dense(self.num_heads * self.out_features)
        self._mix_heads_layer = linen.Dense(self.out_features)

    @linen.compact
    def __call__(self, inputs: Batched[Vector], padding_mask: Boolean[Vector]) -> Batched[Vector]:
        """Compute the attention layer forward pass.

        Note:
            This function will apply a causal mask to its inputs automatically.

        Args:
            inputs (Batched[Vector]): Array containing the input vectors to process.
            padding_mask (Boolean[Vector]): Padding mask to apply in the attention.

        Returns:
            Batched[Vector]: Result of the attention layer for each input.
        """
        # Create the query, key and value vectors
        q_array = self._q_layer(inputs).reshape(-1, self.num_heads, self.inner_features)
        k_array = self._k_layer(inputs).reshape(-1, self.num_heads, self.inner_features)
        v_array = self._v_layer(inputs).reshape(-1, self.num_heads, self.out_features)

        # Generate the optional mask to use in the attention
        causal_mask = create_causal_attention_mask(inputs.shape[0])
        padding_mask = create_padding_attention_mask(padding_mask)

        # Compute the attention applying the mask and concatenate the attention heads
        attention = _multi_head_attention(q_array, k_array, v_array, causal_mask * padding_mask)
        attention = attention.transpose(1, 0, 2)
        attention = attention.reshape(-1, self.num_heads * self.out_features)
        return self._mix_heads_layer(attention)


def compute_attention(
    queries: Batched[Vector],
    keys: Batched[Vector],
    values: Batched[Vector],
    mask: Boolean[Matrix] | None,
) -> Batched[Vector]:
    r"""Compute the attention between queries, keys and values of a single attention head.

    Note:
        This function implements :math:`\rm{softmax}\big(\frac{QK^T}{\sqrt{d_K}})V` for a single
        attention head.

    Args:
        queries (Batched[Vector]): Array containing a batch of query vectors.
        keys (Batched[Vector]): Array containing a batch of key vectors.
        values (Batched[Vector]): Array containing a batch of value vectors.
        mask (Boolean[Matrix] | None): Optional array containing the mask to use in attention.

    Returns:
        Batched[Vector]: Array containing the result of the attention.
    """
    raw_scores = queries @ jnp.matrix_transpose(keys)
    raw_scores = raw_scores / jnp.sqrt(keys.shape[1])
    if mask is not None:
        raw_scores = raw_scores * mask
    return jax.nn.softmax(raw_scores, axis=-1) @ values


# Vectorised version of ``compute_attention`` over multi-head dimension
_multi_head_attention = jax.vmap(compute_attention, in_axes=(1, 1, 1, None))


def create_causal_attention_mask(sequence_length: int) -> Matrix:
    """Create an attention mask for a given sequence length.

    Note:
        The attention mask contains ``1.0`` for unmasked values and ``ENSURE_MASKED_IS_ZERO`` for
        masked values. ``ENSURE_MASKED_IS_ZERO == -1e9`` ensures that the probability of that value
        in a softmax calculation is effectively zero: ``exp(ENSURE_MASKED_IS_ZERO) -> 0.0``.

    Args:
        sequence_length (int): Length of the sequence.

    Returns:
        Matrix: Mask to use in the attention mechanism.
    """
    lower_diagonal_idx = (jnp.arange(1, sequence_length), jnp.arange(0, sequence_length - 1))

    mask = jnp.full((sequence_length, sequence_length), ENSURE_MASKED_IS_ZERO)
    mask = mask.at[jnp.diag_indices(sequence_length)].set(1.0)
    mask = mask.at[lower_diagonal_idx].set(1.0)
    return mask


def create_padding_attention_mask(padding_mask: Boolean[Vector]) -> Matrix:
    """Create the padding attention mask for a given padding mask.

    Note:
        The attention mask contains ``1.0`` for unmasked values and ``ENSURE_MASKED_IS_ZERO`` for
        masked values. ``ENSURE_MASKED_IS_ZERO == -1e9`` ensures that the probability of that value
        in a softmax calculation is effectively zero: ``exp(ENSURE_MASKED_IS_ZERO) -> 0.0``.

    Args:
        padding_mask (Boolean[Vector]): Vector containing the boolean padding mask.

    Returns:
        Matrix: Mask to use in the attention mechanism.
    """
    padding_mask = jnp.invert(padding_mask)[None]
    return (padding_mask * padding_mask.T) * ENSURE_MASKED_IS_ZERO + 1.0
