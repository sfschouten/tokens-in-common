import itertools
from typing import Callable, TypeVar

from tokenizers import Encoding

from tokens_in_common.multitext import MultiText
from tokens_in_common.utils import Reference

T = TypeVar('T')


def tokenize_multitext(multitext: MultiText[str], tokenize_fn: Callable[[str], Encoding]) -> MultiText[list[int]]:
    """
    :param multitext:
    :param tokenize_fn:
    :return:
    """
    leaf_ancestries = list(multitext.get_leaf_ancestries())
    vertex_id_positions = {id(v): v.position for v in multitext.vertices}

    successor_dict = {
        id(a): id(b)
        for i, elements in enumerate(leaf_ancestries)
        for a, b in itertools.pairwise(elements)
    }

    all_token_vertices = {}
    for leaf_i, vertices in enumerate(leaf_ancestries):
        # string_refs, positions, ancestry_hashes = zip(*vertex_elements)

        # join MultiText strings in leaf ancestry into a single string
        full_string = "".join(v.component.value for v in vertices)

        # keep track of original string references that each character came from
        char_vertices = sum(
            ([v] * len(v.component.value) for v in vertices), start=[]
        )

        # tokenize joined string
        encoding = tokenize_fn(full_string)

        new_token_vertices = {}
        for token_index, token_id in enumerate(encoding.ids):
            char_offsets = encoding.token_to_chars(token_index)
            if char_offsets is None:
                # token doesn't correspond to any characters
                if token_index == 0:
                    first = char_vertices[0]
                    assert id(first) not in new_token_vertices
                    new_token_vertices[id(first)] = Reference([token_id])
                else:
                    # TODO: deal with other tokens that don't correspond to any characters?
                    raise NotImplementedError
            else:
                # get the string references and their positions that went into this token
                start, end = char_offsets
                token_vertices = {id(v): v for v in char_vertices[start:end]}

                # select the last (by position) of the string references
                last = max(token_vertices.values(), key=lambda x: x.position)

                if len(token_vertices) == 1:
                    # this token originated from a single string reference
                    if id(last) not in new_token_vertices:
                        new_token_vertices[id(last)] = Reference([])
                    new_token_vertices[id(last)].value.append(token_id)
                else:
                    # this token originated from multiple string references
                    token_vertices_s = sorted(token_vertices.values(), key=lambda x: x.position)

                    # check the connectivity of the vertices contributing to this token
                    connected = [True] + [w in v.children for v, w in itertools.pairwise(token_vertices_s)]
                    cutoff = ([i for i, t in enumerate(connected) if not t] + [len(connected)])[0]

                    # if there is a discontinuity, assert that the nr. of siblings post discontinuity is always 1
                    assert all(len(v.siblings) == 0 for v in token_vertices_s[cutoff:])

                    # assign this token to the last vertex that isn't disconnected from the first
                    assign_id = id(token_vertices_s[:cutoff][-1])
                    if assign_id not in new_token_vertices:
                        new_token_vertices[assign_id] = Reference([])
                    new_token_vertices[assign_id].value.append(token_id)

                    # if other vertices have not been seen yet, initialize them in case we don't see them again
                    for vertex in token_vertices_s:
                        if id(vertex) not in new_token_vertices:
                            new_token_vertices[id(vertex)] = Reference([])

        # add new token vertices to those collected from previous leaf ancestries
        for vertex_id, token_list_ref \
                in sorted(new_token_vertices.items(), key=lambda x: vertex_id_positions[x[0]]):
            if vertex_id not in all_token_vertices:
                # unseen string reference, add it and continue
                all_token_vertices[vertex_id] = token_list_ref
                continue

            cur = all_token_vertices[vertex_id].value
            new = token_list_ref.value
            if cur == new:
                # current and new tokenization are already the same, continue
                continue

            if len(cur) == len(new):
                # the same string reference was tokenized differently in two branches
                # diff_idx = set(i for i in range(len(cur)) if cur[i] != new[i])
                # if 0 in diff_idx:
                #     all_token_vertices[()]
                # elif len(cur) - 1 in diff_idx:
                raise NotImplementedError
            else:
                # the tokenization of two branches has yielded different results for their common prefix
                short = cur if len(cur) < len(new) else new
                long = cur if len(cur) > len(new) else new

                assert long[:len(short)] == short, "Somehow the tokenization did not just change at the end?"

                if new == short:
                    # set common subsequence as new value of this vertex
                    all_token_vertices[vertex_id].value = short

                    # add part that they do not have in common to 'next' vertex
                    successors = [successor_dict[vertex_id] for j in range(0, leaf_i)]
                    for successor in successors:
                        assert successor in all_token_vertices, \
                            "Somehow the successor vertex of a previous leaf ancestry does not exist yet?"
                        for x in long[len(short):]:
                            all_token_vertices[successor].value.insert(0, x)
                else:  # old == short
                    # add part that they do not have in common to 'next' vertex
                    successor = successor_dict[vertex_id]
                    assert successor not in all_token_vertices, \
                        "Somehow the current vertex's successor already exists?"
                    for x in long[len(short):]:
                        new_token_vertices[successor].value.insert(0, x)

    return multitext.copy(new_component_map=all_token_vertices)
