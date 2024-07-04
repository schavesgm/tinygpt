"""Module containing functionality related to the dataset."""

__all__ = ["VOCABULARY", "VOCABULARY_SIZE", "ONE_HOT_INDEX"]

# Complete vocabulary of the application: `[A-Z^$]`
VOCABULARY: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["^", "$"]

# Size of the vocabulary
VOCABULARY_SIZE: int = len(VOCABULARY)

# Mapping containing the one-hot index of each entry in the vocabulary
ONE_HOT_INDEX: dict[str, int] = {token: index for index, token in enumerate(VOCABULARY)}
