from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar('T')


@dataclass
class Reference(Generic[T]):
    """
    Helper class to store and pass around references to primitives and/or mutable types.
    """
    value: T

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return id(self)


def list_find(lst, sub_lst):
    for i in range(len(lst)):
        if lst[i:i+len(sub_lst)] == sub_lst:
            return i
    raise ValueError

