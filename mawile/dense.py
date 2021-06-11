from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.player import Player
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu, swish
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

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


def reward_computing_helper(
    battle: AbstractBattle,
    weight_victory: float = 30.0,
    weight_fainted: float = 5.0,
    weight_health: float = 5.0,
    weight_status: float = 1.0,
    weight_boosts: float = 1.0,
) -> float:
    def evaluate_unit(unit: Pokemon) -> float:
        score_health = weight_health * unit.current_hp_fraction
        score_boosts = weight_boosts * sum(unit.boosts.values())

        score_fainted = -weight_fainted if unit.fainted else 0
        score_status = -weight_status if unit.status else 0

        return score_health + score_fainted + score_status + score_boosts

    score = +weight_victory if battle.won else -weight_victory if battle.lost else 0
    score += sum(map(evaluate_unit, battle.team.values()))
    score -= sum(map(evaluate_unit, battle.opponent_team.values()))

    return score


@dataclass
class DenseQPlayer(MemoryPlayer[np.array, np.array]):
    """
    An agent trained with Deep Q-learning.
    """

    model: Sequential = field(default_factory=init_default_model)
    exploration_rate: float = 0.95

    INPUT_SIZE: ClassVar[int] = 6772
    ACTION_SPACE: ClassVar[int] = 22

    def __post_init__(self):
        Player.__init__(self, max_concurrent_battles=0)

    def battle_to_score(self, battle: AbstractBattle) -> float:
        return reward_computing_helper(battle)

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
