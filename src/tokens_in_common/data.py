import itertools
from enum import Enum
from typing import List, Union, Tuple

from tokens_in_common.multitext import MultiText
from tokens_in_common.utils import Reference


class OptionStringBuildMode(Enum):
    FULL = 1
    STANDARD = 2
    FRUGAL = 3


def multitext_from_option_strings(mode: OptionStringBuildMode, text: List[Union[str, Tuple[str]]]) -> MultiText[str]:
    """
    A function to build the MultiText representation using a list of 'option strings'.
    :param mode:
    :param text: A list of parts which are either a string or a tuple of strings, where the latter represent a set
    of optional strings that each could occupy the next part of the text.
    """
    # wrap each string
    text_fragments: List[Tuple[Reference]] = []
    for part in text:
        if isinstance(part, str):
            text_fragments.append((Reference(part),))
        elif isinstance(part, tuple):
            text_fragments.append(tuple(Reference(f) for f in part))

    fragment_refs: List[Tuple[Reference, int]] = []
    parent_indices = []
    if mode == OptionStringBuildMode.FULL:
        # include every possible combination separately
        for i, path in enumerate(itertools.product(*text_fragments)):
            fragment_refs.extend([(x, i) for i, x in enumerate(path)])
            parent_indices.extend([[]] + [[i * len(path) + j] for j in range(len(path)-1)])
    elif mode == OptionStringBuildMode.STANDARD:
        # build a tree, with a branching point for each fragment with more than one option
        current_expansion = 1  # how many branches there are
        for pos, part in enumerate(text_fragments):
            # iterate over the vertices added in the last iteration
            for j in range(len(fragment_refs) - current_expansion, len(fragment_refs)):
                for f in part:
                    fragment_refs.append((f, pos))
                    if j >= 0:
                        parent_indices.append([j])
                    else:
                        parent_indices.append([])

            current_expansion *= len(part)
    elif mode == OptionStringBuildMode.FRUGAL:
        # first process the singletons: the fragments that have no alternatives
        singleton_indices = {}
        for pos, fragment in enumerate(text_fragments):
            if len(fragment) == 1:
                singleton_indices[pos] = len(fragment_refs)
                fragment_refs.append((fragment[0], pos))
                if pos - 1 in singleton_indices:
                    parent_indices.append([pos - 1])
                else:
                    parent_indices.append([])

        current_expansion = 1
        for pos, part in enumerate(text_fragments):
            if len(part) == 1:
                continue
            for j in range(len(fragment_refs) - current_expansion, len(fragment_refs)):
                for f in part:
                    fragment_refs.append((f, pos))
                    parents = []
                    if pos - 1 in singleton_indices:
                        parents.append(singleton_indices[pos - 1])
                    if j >= len(singleton_indices):
                        parents.append(j)
                    parent_indices.append(parents)

            current_expansion *= len(part)

    return MultiText.from_vertex_elements(fragment_refs, parent_indices)
