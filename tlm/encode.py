"""Module containing functionality to encode sequences."""

import jax.numpy as jnp

from .data import ONE_HOT_INDEX, VOCABULARY_SIZE
from .types import Batched, Vector

__all__ = ["encode_sequence", "positionally_encode"]


def positionally_encode(vectors: Batched[Vector], constant: float = 10_000.0) -> Batched[Vector]:
    r"""Return the positionally encoded version of a batch of input vectors.

    Note:
        This function implements the positional encoding strategy of the transformers. As a result,
        if generate a matrix of the same shape as ``vectors`` whose entries follow the following
        patterns:

        .. math::

            P_{p, 2 i} = \sin\big( \frac{p}{A^{2i \over d}} \big),
            P_{p, 2 i + 1} = \cos\big( \frac{p}{A^{2i \over d}} \big),

        where :math:`d` is the dimension of the input vectors, :math:`A` is a constant, :math:`p`
        encodes the position of each vector in the batch and :math:`i` encodes the position of each
        dimension in each vector.

    Args:
        vectors (Batched[Vector]): Vectors to encode.
        constant (float): Constant to use in the positional encoding formula.

    Returns:
        Batched[Vector]: Positionally encoded vectors.
    """
    batch_size: int = vectors.shape[0]
    embeddings: int = vectors.shape[1]

    # Compute the position indices (in the sequence), the embedding arguments (2 * i / embeddings)
    # and exp-log of the argument
    positions = jnp.arange(0, batch_size)[:, None]
    embed_arg = jnp.arange(0, embeddings, 2) / embeddings
    arguments = jnp.exp(jnp.log(positions) - embed_arg * jnp.log(10_000))

    positional_encodings = jnp.zeros_like(vectors)
    positional_encodings = positional_encodings.at[:, 0::2].set(jnp.sin(arguments))
    positional_encodings = positional_encodings.at[:, 1::2].set(jnp.cos(arguments))

    return vectors + positional_encodings


def encode_sequence(sequence: str, batch_size: int) -> Batched[Vector]:
    """Return the encoded version of a given sequence.

    Note:
        This function will pad the batch of vectors with zeroes to match the correct batch size.

    Args:
        sequence (str): Sequence to encode.
        batch_size (int): Target batch size to produce.

    Returns:
        Batched[Vector]: Array containing a batch of encoding vectors.
    """
    one_hot_encodings = jnp.vstack([_one_hot_encode(token)[None] for token in sequence])
    return _pad_vectors(one_hot_encodings, batch_size)


def _pad_vectors(vectors: Batched[Vector], batch_size: int) -> Batched[Vector]:
    """Return a padded version of the input sequence to the correct batch size.

    Note:
        The padded entries will contain vectors of zeroes.

    Args:
        vectors (Batched[Vector]): Collection of vectors to pad.
        batch_size (int): Number of elements in the batch.

    Returns:
        Batched[Vector]: Array containing a batch of padded encoding vectors.
    """
    num_entries: int = vectors.shape[0]
    if num_entries > batch_size:
        raise ValueError(f"Input array is larger than provided {batch_size=}")
    return jnp.pad(vectors, ((0, max(0, batch_size - num_entries)), (0, 0)))


def _one_hot_encode(token: str) -> Vector:
    """Return the encoded version of a given token.

    Args:
        token (str): Token to encode.

    Returns:
        Vector: Array containing the one-hot encoding of the input token.

    Raises:
        ValueError: if ``token`` is not a valid string.
    """
    if token.upper() not in ONE_HOT_INDEX:
        raise ValueError(f"{token=} is not a valid token.")
    one_hot = jnp.zeros((VOCABULARY_SIZE,))
    one_hot = one_hot.at[ONE_HOT_INDEX[token]].set(1.0)
    return one_hot
