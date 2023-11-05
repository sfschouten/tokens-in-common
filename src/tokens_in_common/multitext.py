from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Optional, TypeVar, Generic
from dataclasses import dataclass, field

import numpy as np

from tokens_in_common.utils import Reference

T = TypeVar('T')
U = TypeVar('U')
Component = Optional[Reference[Mapping[int, T]]]
VertexID = int


@dataclass
class MultiText(Generic[T]):
    """
    A datastructure used to represent texts that have (large) parts in common.
    Based on a DAG, but with additional positional information for each vertex.
    """

    @dataclass
    class Vertex:
        component: Component
        position: int

        parents: list["MultiText.Vertex"] = field(default_factory=list)
        children: list["MultiText.Vertex"] = field(default_factory=list)

        @property
        def siblings(self):
            result = sum((p.children for p in self.parents), start=[])
            if self in result:
                result = result.remove(self)
            return result

        def get_ancestry(self, include_self=False):
            result = []
            for parent in self.parents:
                result.extend(parent.get_ancestry(include_self=True))
            if include_self:
                result.append(self)
            return result

        def calc_ancestral_component_hash(self):
            ancestry = sorted(self.get_ancestry(include_self=True), key=lambda v: v.position)
            hashable = sum((tuple(v.component.value) for v in ancestry), start=tuple())
            return hash(hashable)

        def get_descendants(self, include_self=False):
            result = []
            for child in self.children:
                result.extend(child.get_descendants(include_self=True))
            if include_self:
                result.append(self)
            return result

        def get_weakly_connected_component(self):
            visited = set()
            result = []
            queue = [self]
            while queue:
                v = queue.pop(0)
                result.append(v)
                for w in v.children + v.parents:
                    if id(w) not in visited:
                        visited.add(id(w))
                        queue.append(w)
            return result

    _vertices: list[Vertex] = field(default_factory=list)
    _arcs: list[tuple[Vertex, Vertex]] = field(default_factory=list)

    @property
    def vertices(self):
        for v in self._vertices:
            yield v

    @property
    def arc_components(self):
        for (a, b) in self._arcs:
            yield a.component, b.component

    def __post_init__(self):
        for parent, child in self._arcs:
            parent.children.append(child)
            child.parents.append(parent)

        # TODO check the following
        #  - acyclic
        #  - ancestry of any node cannot contain more than one node per position

    @classmethod
    def from_vertex_elements(cls, elements: list[tuple[Reference[T], int]], parent_indices: list[list[int]]):
        """
        :param elements: A list of vertex elements.
        :param parent_indices: For each element a list with the indices of its parents.
        """
        vertices = [cls.Vertex(f, i) for f, i in elements]
        arcs = list((vertices[b], vertices[a]) for a, parents in enumerate(parent_indices) for b in parents)
        return MultiText(_vertices=vertices, _arcs=arcs)

    def get_leaf_ancestries(self) -> Iterator[list["MultiText.Vertex"]]:
        leafs = [v for v in self._vertices if len(v.children) == 0]
        for leaf in leafs:
            result = leaf.get_ancestry(include_self=True)
            sorted_vertices = sorted(result, key=lambda c: c.position)
            yield sorted_vertices

    def add_vertex(self, component, position, parents=None, children=None):
        new_vertex = MultiText.Vertex(component, position, parents, children)
        self._vertices.append(new_vertex)
        if children is not None:
            self._arcs.extend((new_vertex, child) for child in children)
        if parents is not None:
            self._arcs.extend((parent, new_vertex) for parent in parents)

    def copy(self, new_component_map: Optional[Mapping[VertexID, Reference[U]]]) -> "MultiText[U]":
        """
        Creates a copy of this MultiText, optionally replacing the components in the vertices.
        :param new_component_map:
        :return:
        """
        if new_component_map is None:
            new_component_map = {id(v): v.component for v in self._vertices}
        vertices = {
            id(v): self.Vertex(new_component_map[id(v)], v.position)
            for v in self._vertices
        }
        arcs = [(vertices[id(a)], vertices[id(b)]) for a, b in self._arcs]
        return MultiText(_vertices=list(vertices.values()), _arcs=arcs)

    class PositioningMethod(Enum):
        FULL_ALIGNMENT = 0
        NO_ALIGNMENT = 1

    def prepare_inputs(
            self, pos_method=PositioningMethod.FULL_ALIGNMENT
    ) -> tuple[list[int], list[int], list[list[bool]], list[tuple[Component, int, int]]]:
        """
        Prepares the MultiText for input into a language model.
        :param pos_method: how to assign the tokens their position_ids
        :return: input_ids, position_ids, attention_mask, vertex elements (the component, vertex position, and
        ancestral hash of each token).
        """
        # get root vertices
        roots = [v for v in self._vertices if len(v.parents) == 0]

        tokens = []
        token_pos_ids = []
        token_vertex_elements = []

        total_nr_elements = sum(len(v.component.value) for v in self._vertices)
        attention_mask = [[False] * total_nr_elements for _ in range(total_nr_elements)]
        nr_vertex_positions = len(set(v.position for v in self.vertices))
        idx = {}
        for root in roots:
            if pos_method == MultiText.PositioningMethod.FULL_ALIGNMENT:
                # first calculate the maximum length per vertex position in this weakly connected component
                pos_max_length = [0] * nr_vertex_positions
                for v in root.get_weakly_connected_component():
                    current_max = pos_max_length[v.position]
                    if len(v.component.value) > current_max:
                        pos_max_length[v.position] = len(v.component.value)
                pos_starts = [0] + list(np.cumsum(pos_max_length))

            # traverse this root's subgraph
            queue: list[tuple[MultiText.Vertex, MultiText.Vertex]] = [(root, None)]
            while queue:
                v, parent = queue.pop(0)
                parent_last_idx = idx[id(parent)][1] - 1 if parent is not None else -1
                length = len(v.component.value)

                start = {   # where to start for the position_ids of this vertex
                    MultiText.PositioningMethod.FULL_ALIGNMENT:
                        pos_starts[v.position],
                    MultiText.PositioningMethod.NO_ALIGNMENT:
                        token_pos_ids[parent_last_idx] if parent is not None else 0,
                }[pos_method]

                if id(v) not in idx:
                    # unseen vertex, extend `tokens`, `token_pos_ids` and `token_vertex_elements`
                    idx[id(v)] = (len(tokens), len(tokens) + length)
                    tokens.extend(v.component.value)
                    token_pos_ids.extend(range(start, start + length))
                    token_vertex_elements.extend(
                        [(v.component, v.position, v.calc_ancestral_component_hash())] * length
                    )

                start_idx, end_idx = idx[id(v)]
                for i in range(start_idx, end_idx):
                    # token pays attention to:
                    #  (1) the same things as the last token of this vertex's parent; and
                    if parent is not None:
                        attention_mask[i] = [
                            a or b for a, b in zip(attention_mask[i], attention_mask[parent_last_idx])
                        ]
                    #  (2) the previous tokens within this vertex
                    for j in range(start_idx, i + 1):
                        attention_mask[i][j] = True

                for child in v.children:
                    queue.append((child, v))

        return tokens, token_pos_ids, attention_mask, token_vertex_elements

    def is_causal(self):
        """
        :return: true if it has no arcs going backwards, false otherwise.
        """
        raise NotImplementedError

    def is_multitree(self):
        """
        :return: true if at most one path between any two vertices, false otherwise.
        """
        raise NotImplementedError

    def is_tree(self):
        """
        :return:
        """
        raise NotImplementedError

    def __add__(self, other: "MultiText"):
        vertices = self._vertices + other._vertices
        arcs = self._arcs + other._arcs
        return MultiText(_vertices=vertices, _arcs=arcs)

    def render_with_graphviz(self, name, label_fn=str, **kwargs):
        import graphviz

        g = graphviz.Digraph(name=name)
        g.attr(bgcolor='#ffffff00')

        vertex_positions = sorted(set(v.position for v in self.vertices))
        pos_vertices = [[] for _ in range(len(vertex_positions))]
        for vertex in self._vertices:
            pos_vertices[vertex.position].append(vertex)

        for pos in vertex_positions:
            with g.subgraph(name=f'cluster_{pos}') as c:
                c.attr(label=str(pos), labeljust='l', fontsize='18', bgcolor='#e6e6e640', penwidth='0')
                for vertex in pos_vertices[pos]:
                    c.node(str(id(vertex)), label=label_fn(vertex.component.value))

        for vertex_a, vertex_b in self._arcs:
            g.edge(str(id(vertex_a)), str(id(vertex_b)))

        g.render(**kwargs)
