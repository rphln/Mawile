import asyncio
import logging
import os
from collections import Counter, defaultdict
from pprint import pprint
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from mawile.dense import DenseQPlayer, init_default_model
from mawile.players import NaivePlayer, forget, memory_to_transitions

LOG_LEVEL = os.environ.get("LOG_LEVEL", logging.WARNING)
MODEL_PATH = os.environ.get("MODEL_PATH", "var/checkpoint.keras")

GAMMA = 0.95
BATCH_SIZE = 50_000


def memory_to_dataset(model, memory) -> Tuple[np.array, np.array]:
    transitions = memory_to_transitions(memory)

    x_train = []
    y_train = []

    batch_q_values = model.predict(
        np.vstack([transition.state for transition in transitions])
    )
    batch_q_values_next = model.predict(
        np.vstack([transition.state_next for transition in transitions])
    )

    for (transition, q_values, q_values_next) in zip(
        transitions, batch_q_values, batch_q_values_next
    ):
        state, action, reward, state_next, is_terminal = transition

        if is_terminal:
            q_update = reward
        else:
            q_update = reward + GAMMA * np.amax(q_values_next)

        q_values[action] = q_update

        x_train.append(state)
        y_train.append(q_values)

    return np.vstack(x_train), np.vstack(y_train)


async def main():
    logging.disable(LOG_LEVEL)

    shared_memory = defaultdict(list)
    shared_model = init_default_model()

    try:
        shared_model.load_weights(MODEL_PATH)
    except:
        print("Using a fresh model.")
    else:
        print("Using the saved model.")

    checkpoint = ModelCheckpoint(MODEL_PATH, save_weights_only=True)
    early_stop = EarlyStopping(monitor="loss", patience=5)

    players = [
        NaivePlayer(max_concurrent_battles=0),
        DenseQPlayer(shared_memory, shared_model),
        DenseQPlayer(shared_memory, shared_model),
        DenseQPlayer(shared_memory, shared_model),
        DenseQPlayer(shared_memory, shared_model),
        DenseQPlayer(shared_memory, shared_model),
        RandomPlayer(max_concurrent_battles=0),
    ]

    statistics = Counter()

    # Round-robin training.
    while True:
        cross_evaluation = await cross_evaluate(players, n_challenges=1)
        forget(shared_memory, retain=200)

        statistics += {
            (first, second): win_rate or 0
            for first, matches in cross_evaluation.items()
            for second, win_rate in matches.items()
        }

        pprint(statistics)

        x, y = memory_to_dataset(shared_model, shared_memory)
        shared_model.fit(x, y, epochs=100, callbacks=[checkpoint, early_stop])


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
