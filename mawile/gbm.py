from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from mawile.encode import encode_moves, encode_unit, encode_units
from mawile.players import MemoryPlayer, TAction


def init_default_model():
    model = MultiOutputRegressor(
        GradientBoostingRegressor(max_depth=22, max_features="log2", verbose=1)
    )
    model.fit(
        np.random.randn(2, GradientBoostingPlayer.INPUT_SIZE),
        np.random.randn(2, GradientBoostingPlayer.ACTION_SPACE),
    )

    return model


@dataclass
class GradientBoostingPlayer(MemoryPlayer[np.array, np.array]):
    """
    An agent trained with `GradientBoostingRegressor`.
    """

    model: MultiOutputRegressor = None

    exploration_rate: float = 0.05

    INPUT_SIZE: ClassVar[int] = 16726
    ACTION_SPACE: ClassVar[int] = 22

    def __post_init__(self):
        super().__post_init__()

    def battle_to_state(self, battle: AbstractBattle) -> np.array:
        moves = np.hstack(
            encode_moves(battle.available_moves, battle.opponent_active_pokemon)
        )

        player = encode_unit(battle.active_pokemon)
        opponent = encode_unit(battle.opponent_active_pokemon)

        team = encode_units(battle.available_switches).flatten()

        player_remaining = np.count_nonzero(
            unit.fainted for unit in battle.team.values()
        )
        opponent_remaining = np.count_nonzero(
            unit.fainted for unit in battle.opponent_team.values()
        )

        return np.hstack(
            [player, opponent, team, player_remaining, opponent_remaining, moves]
        )

    def state_to_action(self, state: np.array, battle: AbstractBattle) -> np.array:
        if np.random.uniform() < self.exploration_rate:
            return np.random.randint(self.ACTION_SPACE)

        (q_values,) = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values)

    def action_to_move(self, action: TAction, battle: AbstractBattle) -> BattleOrder:
        return Gen8EnvSinglePlayer._action_to_move(self, action, battle)
