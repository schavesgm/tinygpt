"""Module containing functionality to encode sequences."""

import random
from collections.abc import Generator
from pathlib import Path

import jax.numpy as jnp

from .types import Batched, Boolean, Vector

__all__ = ["Dataset", "encode_sequence", "pad_sequence", "positionally_encode"]

# Define the padding and dummy tokens
PADDING_TOKEN: str = "*"
DUMMY_TOKEN: str = "\\"


class Dataset:
    """Class acting a container for a dataset."""

    def __init__(self, in_file: Path | str, batch_size: int) -> None:
        """Instantiate a ``Dataset`` object."""
        self._in_file: Path = Path(in_file)
        self._batch_size: int = batch_size

        with open(self._in_file, encoding="utf-8") as file_stream:
            self._entries: list[str] = file_stream.readlines()
            self._entries = [entry.replace("\n", "").strip() for entry in self._entries]

        unique_entries: set[str] = set()
        for sequence in self._entries:
            for token in sequence:
                unique_entries.add(token)
        self._vocabulary: list[str] = [PADDING_TOKEN] + list(unique_entries)

        # Add another dummy entry to make the vocabulary even (for positional encoding)
        if len(self._vocabulary) % 2 != 0:
            self._vocabulary += [DUMMY_TOKEN]

    def __len__(self) -> int:
        """Return the number of entries in the ``Dataset`` object."""
        return len(self._entries)

    def __iter__(self) -> Generator[str, None, None]:
        """Return an iterator over shuffled sequences in the dataset."""
        for entry in random.sample(self._entries, k=len(self)):
            yield entry[: self._batch_size]

    @property
    def vocabulary(self) -> list[str]:
        """Return the vocabulary in the dataset."""
        return self._vocabulary

    @property
    def one_hot_mapping(self) -> dict[str, int]:
        """Return the mapping to one-hot encode each entry in the dataset."""
        return {token: index for index, token in enumerate(self._vocabulary)}


def pad_sequence(sequence: str, max_length: int) -> str:
    """Return a padded version of the input sequence.

    Note:
        This function will pad the batch of vectors with empty characters "*" to match the correct
        maximum number of tokens.

    Args:
        sequence (str): Sequence to encode.
        max_length (int): Maximum sequence length.

    Returns:
        str: Padded version of the input sequence.
    """
    return sequence + "".join([PADDING_TOKEN] * max(0, max_length - len(sequence)))


def encode_sequence(
    sequence: str, one_hot_mapping: dict[str, int]
) -> tuple[Batched[Vector], Boolean[Vector]]:
    """Return the encoded version of a given sequence.

    Args:
        sequence (str): Sequence to encode.
        one_hot_mapping (dict[str, int]): Mapping to encode each token into a number.

    Returns:
        tuple[Batched[Vector], Boolean[Vector]]: Tuple containing a batch of encoding
            vectors and a batch of mask representing the unpadded tokens.
    """
    encodings = jnp.vstack([_one_hot_encode(token, one_hot_mapping)[None] for token in sequence])
    return encodings, jnp.array([token == PADDING_TOKEN for token in sequence], dtype=jnp.bool)


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


def decode_sequence(indices: Vector, one_hot_mapping: dict[int, str]) -> str:
    """Return a decoded version of an input collection of tokens.

    Args:
        indices (Vector): Vector containing the indices to select for each token.
        one_hot_mapping (dict[str, int]): Mapping to decode each number into a token.

    Returns:
        str: Decoded version of the input sequence.
    """
    return "".join(one_hot_mapping[int(idx)] for idx in indices)


def _one_hot_encode(token: str, one_hot_mapping: dict[str, int]) -> Vector:
    """Return the encoded version of a given token.

    Args:
        token (str): Token to encode.
        one_hot_mapping (dict[str, int]): Mapping to encode each token into a number.

    Returns:
        Vector: Array containing the one-hot encoding of the input token.

    Raises:
        ValueError: if ``token`` is not a valid string.
    """
    if token.upper() not in one_hot_mapping:
        raise ValueError(f"{token=} is not a valid token.")
    one_hot = jnp.zeros((len(one_hot_mapping),))
    one_hot = one_hot.at[one_hot_mapping[token]].set(1.0)
    return one_hot
