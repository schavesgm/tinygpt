"""Small script containing a custom implementation of a transformer."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import random

from tlm.dataset import (
    PADDING_TOKEN,
    Dataset,
    decode_sequence,
    encode_sequence,
    pad_sequence,
    positionally_encode,
)
from tlm.gpt import TinyGPT
from tlm.log import get_logger
from tlm.types import Batched, Boolean, Parameters, Scalar, Vector

LOGGER = get_logger(__file__)


def get_sequence_identity(sequence_1: str, sequence_2: str) -> float:
    """Return the sequence identity between two sequences.

    Note:
        The sequence identity is defined as the number of equal tokens between both sequences
        divided by the total number of tokens in each sequence. The sequences must be of the same
        size as no sequence alignment is performed.

    Args:
        sequence_1 (str): First sequence to compare.
        sequence_2 (str): Second sequence to compare.

    Returns:
        float: Sequence identity between both input sequences.

    Raises:
        ValueError: If ``sequence_1`` and ``sequence_2`` do not have the same length.
    """
    equal_tokens = [t1 == t2 for t1, t2 in zip(sequence_1, sequence_2, strict=True)]
    return sum(equal_tokens) / len(sequence_1)


@jax.jit
def update_function(
    state: TrainState, one_hot_encodings: Batched[Vector], padding_mask: Boolean[Vector]
) -> tuple[TrainState, Scalar, Batched[Vector]]:
    """Function to update the model parameters given some input data.

    Args:
        state (TrainState): ``TrainState`` object containing the parameters to update.
        one_hot_encodings (Batched[Vector]): One-hot encodings of the batch to process.
        padding_mask (Boolean[Mask]): Mask denoting whether the tokens are real or padded.

    Returns:
        tuple[TrainState, Scalar, Batched[Vector]]: Tuple containing the update ``TrainState``
            object, the loss function value and the predicted logits.
    """

    def _forward_pass(parameters: Parameters) -> tuple[Scalar, Batched[Vector]]:
        """Compute the complete forward pass to produce the logits and the losses."""
        logits = state.apply_fn(
            {"params": parameters}, positionally_encode(one_hot_encodings), padding_mask
        )
        loss = optax.losses.softmax_cross_entropy(logits, one_hot_encodings).mean()
        return loss, logits

    compute_value_and_grads = jax.value_and_grad(_forward_pass, has_aux=True)
    (loss, logits), gradients = compute_value_and_grads(state.params)
    train_state = state.apply_gradients(grads=gradients)
    return train_state, loss, logits


def create_generate_function(
    model: TinyGPT, batch_size: int, one_hot_mapping: dict[str, int]
) -> Callable[[str, Parameters], str]:
    """Create a function to generate sequences auto-regressively.

    Note:
        The generated function does not implement beam-search. As a result, it will only select the
        most probable token conditioned on all previous tokens.

    Note:
        The function does not cache the attention computations for speed. In a real scenario, they
        should be cached for speed up.

    Returns:
        Callable[[str, Parameters], str]: Function to generate data auto-regressively.
    """
    inverse_mapping = {index: token for token, index in one_hot_mapping.items()}
    _predict = jax.jit(
        lambda parameters, encodings, mask: model.apply({"params": parameters}, encodings, mask)
    )

    def _generate(sequence: str, parameters: Parameters) -> str:
        """Generate some sequences using the model."""
        one_hot_encodings, padding_mask = encode_sequence(
            pad_sequence(sequence, batch_size), one_hot_mapping
        )
        logits = _predict(parameters, one_hot_encodings, padding_mask)[len(sequence), :]
        selected_token = inverse_mapping[int(jnp.argmax(jax.nn.softmax(logits)))]
        if selected_token == PADDING_TOKEN or len(sequence) == batch_size:
            return sequence
        return _generate(sequence + selected_token, parameters)

    return _generate


def main() -> None:
    """Entry point of the script."""
    num_epochs: int = 10
    batch_size: int = 6
    num_blocks: int = 4

    dataset = Dataset("./dataset.txt", batch_size=4)
    vocabulary_size: int = len(dataset.vocabulary)

    LOGGER.info("Loaded dataset from 'dataset.txt' file")
    LOGGER.info("Vocabulary size: %d", vocabulary_size)
    LOGGER.info("Number of blocks in the model: %d", num_blocks)
    LOGGER.info("Number of tokens in each sequence: %d", batch_size)
    LOGGER.info("Number of epochs at training: %d", num_epochs)

    LOGGER.info("Creating `TinyGPT` instance")
    model = TinyGPT(num_blocks=num_blocks, vocabulary_size=vocabulary_size)
    parameters = model.init(
        random.key(121212),
        jnp.zeros((batch_size, vocabulary_size)),
        jnp.ones((batch_size,), dtype=jnp.bool),
    )

    train_state = TrainState.create(
        apply_fn=model.apply, params=parameters["params"], tx=optax.adamw(3e-4)
    )

    # Partialised functions to encode and decode sequences
    encode = partial(encode_sequence, one_hot_mapping=dataset.one_hot_mapping)
    decode = partial(decode_sequence, one_hot_mapping=dict(enumerate(dataset.vocabulary)))

    LOGGER.info("Starting training")
    for epoch in range(num_epochs):
        for batch, sequence in enumerate(dataset):
            sequence = pad_sequence(sequence, batch_size)
            one_hot_encodings, padding_mask = encode(sequence)
            train_state, loss, logits = update_function(
                train_state, one_hot_encodings, padding_mask
            )
            predicted = decode(jnp.argmax(jax.nn.softmax(logits, axis=1), axis=1))

            if batch % 20 == 0:
                sequence_identity = get_sequence_identity(sequence, predicted)
                LOGGER.info(
                    "%d.%d - loss=%.4f - (reference, prediction)=(%s, %s) - identity=%.2f ",
                    epoch,
                    batch,
                    loss,
                    sequence,
                    predicted,
                    sequence_identity,
                )

    # Generate the function to create sequence auto-regressively
    generate = create_generate_function(model, batch_size, dataset.one_hot_mapping)

    vocabulary = set(dataset.vocabulary) - {PADDING_TOKEN}
    available_tokens = "".join(vocabulary)
    LOGGER.info("Generation of sequences. Run <Ctrl-C> to interrupt.")
    LOGGER.info("Available tokens: %s", available_tokens)
    while True:
        input_sequence = input("Input sequence: ")

        if len(input_sequence) > batch_size:
            print(f"ERROR: Input sequence must be smaller than context size: {batch_size}")
            continue

        if any(token not in vocabulary for token in input_sequence) and input_sequence != "":
            print(f"ERROR: Input sequence contains unavailable tokens: {available_tokens}")
            continue

        print(generate(input_sequence, train_state.params))


if __name__ == "__main__":
    main()
