from typing import Iterable, Optional

import numpy as np
from poke_env.data import MOVES, POKEDEX
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status
from sklearn.preprocessing import label_binarize

from mawile.utils import pad

ABILITIES = {
    ability for species in POKEDEX.values() for ability in species["abilities"].values()
}


def encode_types(types: Iterable[PokemonType]) -> np.ndarray:
    """
    Converts a type to an one-hot vector representation.
    """

    return label_binarize(
        [variant.name if variant else "" for variant in types],
        classes=[variant.name for variant in PokemonType],
    )


def encode_status(status: Iterable[Status]) -> np.ndarray:
    """
    Converts a status to an one-hot vector representation.
    """

    return label_binarize(
        [variant.name if variant else "" for variant in status],
        classes=[variant.name for variant in Status],
    )


def encode_move_names(moves: Iterable[Optional[str]]) -> np.ndarray:
    """
    Converts a move to an one-hot vector representation.
    """

    return label_binarize(
        [name or "" for name in moves],
        classes=list(MOVES),
    )


def encode_ability(abilities: Iterable[Optional[str]]) -> np.ndarray:
    """
    Converts an ability to an one-hot vector representation.
    """

    return label_binarize(
        [name or "" for name in abilities],
        classes=list(ABILITIES),
    )


def encode_species(species: Iterable[str]) -> np.ndarray:
    """
    Converts a species to an one-hot vector representation.
    """

    return label_binarize(
        list(species),
        classes=list(POKEDEX),
    )


def encode_unit(pokemon: Optional[Pokemon]) -> np.ndarray:
    """
    Converts an unit to a multi-dimensional array representation.
    """

    if not pokemon:
        return np.zeros(1661)

    species = encode_species([pokemon.species])
    ability = encode_ability([pokemon.ability])
    status = encode_status([pokemon.status])

    types = np.sum(
        encode_types([pokemon.type_1, pokemon.type_2]), axis=0, keepdims=True
    )

    boosts = np.fromiter(pokemon.boosts.values(), float).reshape(1, -1)
    scores = np.fromiter(pokemon.base_stats.values(), float).reshape(1, -1)

    return np.hstack((species, scores, types, ability, status, boosts)).flatten()


def encode_units(units: Iterable[Optional[Pokemon]]) -> np.ndarray:
    """
    Converts a sequence of units to a multi-dimensional array representation.
    """

    return np.vstack([encode_unit(unit) for unit in pad(units, to_length=6)])


def encode_move(move: Optional[Move], opponent: Pokemon) -> np.array:
    """
    Converts a move to a multi-dimensional array representation.
    """

    (name,) = encode_move_names([move.id if move else None])
    return name


def encode_moves(moves: Iterable[Move], opponent: Pokemon, count: int = 4) -> np.array:
    """
    Converts a sequence of moves to a multi-dimensional array representation.
    """

    return encode_move_names(
        [move.id if move else None for move in pad(moves, to_length=count)]
    )
