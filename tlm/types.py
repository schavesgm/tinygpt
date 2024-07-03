"""Module containing custom defined type hints."""

from typing import Annotated, TypeVar

import jax

T = TypeVar("T")

__all__ = [
    "Batched",
    "Boolean",
    "Matrix",
    "Vector",
]

# Custom base arrays
type Vector = Annotated[jax.Array, tuple[int]]
type Matrix = Annotated[jax.Array, tuple[int, int]]

# Annotation for arrays containing a first batch dimension
type Batched[T] = Annotated[T, "batched first dimension"]

# Annotation for arrays that can be treated as boolean
type Boolean[T] = Annotated[T, "treated as boolean"]
