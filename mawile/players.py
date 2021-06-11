from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    DefaultDict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
)

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

TState = TypeVar("TState")
TAction = TypeVar("TAction")


Memory: Type[DefaultDict[AbstractBattle, List[Observation[TState, TAction]]]]


class Observation(Generic[TState, TAction], NamedTuple):
    state: TState
    action: Optional[TAction]
    score: float
    is_terminal: bool


@dataclass
class MemoryPlayer(Generic[TState, TAction], Player, ABC):
    memory: Memory = field(default_factory=lambda: defaultdict(list))

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        state = self.battle_to_state(battle)
        score = self.battle_to_score(battle)
        action = self.state_to_action(state, battle)
        move = self.action_to_move(action, battle)

        self.memory[battle].append(Observation(state, action, score, False))

        return move

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        state = self.battle_to_state(battle)
        score = self.battle_to_score(battle)

        self.memory[battle].append(Observation(state, None, score, True))

    @abstractmethod
    def battle_to_score(self, battle: AbstractBattle) -> float:
        raise NotImplementedError()

    @abstractmethod
    def battle_to_state(self, battle: AbstractBattle) -> TState:
        raise NotImplementedError()

    @abstractmethod
    def state_to_action(self, state: TState, battle: AbstractBattle) -> TAction:
        raise NotImplementedError()

    @abstractmethod
    def action_to_move(self, action: TAction, battle: AbstractBattle) -> BattleOrder:
        raise NotImplementedError()

    @classmethod
    def memory_to_transitions(cls, memory: Memory) -> Iterable:
        for battle, steps in memory.items():
            for n, (step, step_next) in enumerate(zip(steps, steps[1:])):
                state, action, score, _ = step
                state_next, _, score_next, is_terminal = step_next

                reward = score_next - score

                yield state, action, reward, state_next, is_terminal

    @classmethod
    def forget(cls, memory: Memory, retain: int) -> List:
        return list(cls.forget_lazy(memory, retain))

    @classmethod
    def forget_lazy(cls, memory: Memory, retain: int) -> Iterable:
        for key in list(memory.keys())[:-retain]:
            yield memory.pop(key)
