import asyncio
import logging
import os
from collections import Counter, defaultdict
from pprint import pprint
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from keras.callbacks import ModelCheckpoint
from poke_env.environment.move import Move
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate

from mawile.dense import DenseQPlayer, init_default_model
from mawile.players import MemoryPlayer

LOG_LEVEL = os.environ.get("LOG_LEVEL", logging.WARNING)
MODEL_PATH = os.environ.get("MODEL_PATH", "var/checkpoint")

GAMMA = 0.95
BATCH_SIZE = 50_000


class GreedyPlayer(Player):
    def choose_move(self, battle):
        if not battle.available_moves:
            return self.choose_random_move(battle)

        def evaluate_move(move: Move) -> float:
            return (
                move.accuracy
                * move.base_power
                * move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
            )

        return self.create_order(max(battle.available_moves, key=evaluate_move))


def memory_to_dataset(model, memory) -> Tuple[np.array, np.array]:
    batch = list(MemoryPlayer.memory_to_transitions(memory))

    x_train = []
    y_train = []

    batch_q_values = model.predict(np.vstack([step[0] for step in batch]))
    batch_next_q_values = model.predict(np.vstack([step[3] for step in batch]))

    for (
        (state, action, reward, state_next, terminal),
        q_values,
        next_q_values,
    ) in zip(batch, batch_q_values, batch_next_q_values):
        if terminal:
            q_update = reward
        else:
            q_update = reward + GAMMA * np.argmax(next_q_values)

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

    checkpoint_callback = ModelCheckpoint(MODEL_PATH, save_weights_only=True)

    players = [
        GreedyPlayer(max_concurrent_battles=0),
        RandomPlayer(max_concurrent_battles=0),
        DenseQPlayer(shared_memory, shared_model, exploration_rate=0.01),
        DenseQPlayer(shared_memory, shared_model, exploration_rate=0.05),
        DenseQPlayer(shared_memory, shared_model, exploration_rate=0.10),
        DenseQPlayer(shared_memory, shared_model, exploration_rate=0.20),
        DenseQPlayer(shared_memory, shared_model, exploration_rate=0.40),
    ]

    pre_train_player = players[2]
    pre_train_against = players[0]

    # Pre-training step.
    for it in range(10):
        pre_train_player.exploration_rate = max(0.05, 0.6 ** it)

        await cross_evaluate([pre_train_player, pre_train_against], n_challenges=10)
        MemoryPlayer.forget(shared_memory, retain=200)

        x_train, y_train = memory_to_dataset(shared_model, shared_memory)
        shared_model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint_callback])

    statistics = Counter()

    # Round-robin training.
    while True:
        cross_evaluation = await cross_evaluate(players, n_challenges=1)
        MemoryPlayer.forget(shared_memory, retain=200)

        statistics += {
            (first, second): win_rate or 0
            for first, matches in cross_evaluation.items()
            for second, win_rate in matches.items()
        }

        pprint(statistics)

        x_train, y_train = memory_to_dataset(shared_model, shared_memory)
        shared_model.fit(x_train, y_train, callbacks=[checkpoint_callback])


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
