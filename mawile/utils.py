from itertools import zip_longest


def pad(iterable, to_length, default_at=lambda n: None):
    """
    Pads `iterable` to the specified `length` with the value returned by `default_at`.
    """

    return (
        item or default_at(n) for n, item in zip_longest(range(to_length), iterable)
    )
