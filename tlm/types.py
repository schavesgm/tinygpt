"""Module containing custom defined type hints."""

from typing import Annotated, TypeVar

import jax

T = TypeVar("T")

__all__ = [
    "Batched",
    "Boolean",
    "Matrix",
    "Parameters",
    "Scalar",
    "Vector",
]

# Custom base arrays
type Scalar = Annotated[jax.Array, tuple[()]]
type Vector = Annotated[jax.Array, tuple[int]]
type Matrix = Annotated[jax.Array, tuple[int, int]]

# Type to the parameters in the model
type Parameters = dict[str, jax.Array]

# Annotation for arrays containing a first batch dimension
type Batched[T] = Annotated[T, "batched first dimension"]

# Annotation for arrays that can be treated as boolean
type Boolean[T] = Annotated[T, "treated as boolean"]
