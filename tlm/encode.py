"""Module containing functionality to encode sequences."""

import jax.numpy as jnp


from .types import Batched, Vector

__all__ = ["encode_sequence", "one_hot_encode", "pad_vectors"]

# Complete vocabulary of the application: `[A-Z^$]`
VOCABULARY: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["^", "$"]

# Size of the vocabulary
VOCABULARY_SIZE: int = len(VOCABULARY)

# Mapping containing the one-hot index of each entry in the vocabulary
ONE_HOT_INDEX: dict[str, int] = {token: index for index, token in enumerate(VOCABULARY)}


def pad_vectors(vectors: Batched[Vector], batch_size: int) -> Batched[Vector]:
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


def encode_sequence(sequence: str) -> Batched[Vector]:
    """Return the encoded version of a given sequence.

    Args:
        sequence (str): Sequence to encode.

    Returns:
        Batched[Vector]: Array containing a batch of encoding vectors.
    """
    return jnp.vstack([one_hot_encode(token)[None] for token in sequence])


def one_hot_encode(token: str) -> Vector:
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
