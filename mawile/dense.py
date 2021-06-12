from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu, swish
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from mawile.common import reward_computing_helper
from mawile.encode import encode_moves, encode_unit
from mawile.players import MemoryPlayer, TAction

LEARNING_RATE = 0.001


def init_default_model():
    model = Sequential()
    model.add(Dense(512, activation=swish, input_dim=DenseQPlayer.INPUT_SIZE))
    model.add(Dense(512, activation=swish))
    model.add(Dense(512, activation=swish))
    model.add(Dense(512, activation=swish))
    model.add(Dense(DenseQPlayer.ACTION_SPACE, activation=relu))

    model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

    return model


@dataclass
class DenseQPlayer(MemoryPlayer[np.array, np.array]):
    """
    An agent trained with Deep Q-learning.
    """

    model: Sequential = field(default_factory=init_default_model)

    exploration_rate: float = 0.05

    INPUT_SIZE: ClassVar[int] = 6772
    ACTION_SPACE: ClassVar[int] = 22

    def __post_init__(self):
        super().__post_init__()

    def battle_to_score(self, battle: AbstractBattle) -> float:
        return reward_computing_helper(
            battle,
            weight_victory=1.0,
            weight_fainted=0.0,
            weight_health=0.0,
            weight_status=0.0,
            weight_boosts=0.0,
        )

    def battle_to_state(self, battle: AbstractBattle) -> np.array:
        moves = encode_moves(battle.available_moves, battle.opponent_active_pokemon)

        (player,) = encode_unit(battle.active_pokemon)
        (opponent,) = encode_unit(battle.opponent_active_pokemon)

        player_remaining = np.count_nonzero(
            unit.fainted for unit in battle.team.values()
        )
        opponent_remaining = np.count_nonzero(
            unit.fainted for unit in battle.opponent_team.values()
        )

        return np.hstack(
            [player, opponent, player_remaining, opponent_remaining, moves]
        )

    def state_to_action(self, state: np.array, battle: AbstractBattle) -> np.array:
        if np.random.uniform() < self.exploration_rate:
            return np.random.randint(self.ACTION_SPACE)

        (q_values,) = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values)

    def action_to_move(self, action: TAction, battle: AbstractBattle) -> BattleOrder:
        return Gen8EnvSinglePlayer._action_to_move(self, action, battle)
