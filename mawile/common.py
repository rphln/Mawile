from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon


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
