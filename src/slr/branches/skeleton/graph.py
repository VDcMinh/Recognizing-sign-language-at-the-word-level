"""Graph topology definitions for reduced skeleton keypoint sets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SELECTED_27_NODE_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb_mid",
    "left_thumb_tip",
    "left_index_mid",
    "left_index_tip",
    "left_middle_mid",
    "left_middle_tip",
    "left_ring_mid",
    "left_ring_tip",
    "left_pinky_mid",
    "left_pinky_tip",
    "right_thumb_mid",
    "right_thumb_tip",
    "right_index_mid",
    "right_index_tip",
    "right_middle_mid",
    "right_middle_tip",
    "right_ring_mid",
    "right_ring_tip",
    "right_pinky_mid",
    "right_pinky_tip",
]

SELECTED_31_NODE_NAMES = [
    *SELECTED_27_NODE_NAMES,
    "mouth_left_corner",
    "mouth_right_corner",
    "upper_lip",
    "lower_lip",
]

SELECTED_27_UNDIRECTED_EDGES = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (5, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (5, 11),
    (11, 12),
    (5, 13),
    (13, 14),
    (5, 15),
    (15, 16),
    (6, 17),
    (17, 18),
    (6, 19),
    (19, 20),
    (6, 21),
    (21, 22),
    (6, 23),
    (23, 24),
    (6, 25),
    (25, 26),
]

SELECTED_31_EXTRA_UNDIRECTED_EDGES = [
    (27, 29),
    (29, 28),
    (28, 30),
    (30, 27),
    (29, 30),
    (0, 29),
    (0, 30),
]

SELECTED_27_INWARD_EDGES = [
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 1),
    (5, 3),
    (4, 2),
    (6, 4),
    (7, 5),
    (8, 7),
    (9, 5),
    (10, 9),
    (11, 5),
    (12, 11),
    (13, 5),
    (14, 13),
    (15, 5),
    (16, 15),
    (17, 6),
    (18, 17),
    (19, 6),
    (20, 19),
    (21, 6),
    (22, 21),
    (23, 6),
    (24, 23),
    (25, 6),
    (26, 25),
]

SELECTED_31_EXTRA_INWARD_EDGES = [
    (27, 29),
    (28, 29),
    (29, 0),
    (30, 29),
    (30, 0),
    (30, 27),
    (30, 28),
]

GRAPH_NODE_NAMES = {
    "selected_27": SELECTED_27_NODE_NAMES,
    "selected_31": SELECTED_31_NODE_NAMES,
}
GRAPH_EDGES = {
    "selected_27": SELECTED_27_UNDIRECTED_EDGES,
    "selected_31": SELECTED_27_UNDIRECTED_EDGES + SELECTED_31_EXTRA_UNDIRECTED_EDGES,
}
GRAPH_INWARD_EDGES = {
    "selected_27": SELECTED_27_INWARD_EDGES,
    "selected_31": SELECTED_27_INWARD_EDGES + SELECTED_31_EXTRA_INWARD_EDGES,
}


def _validate_layout(layout: str) -> str:
    """Validate a supported reduced skeleton layout name."""

    if layout not in GRAPH_NODE_NAMES:
        raise KeyError(f"Unsupported skeleton layout: {layout!r}.")
    return layout


def _validate_edges(num_nodes: int, edges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Validate edge endpoints and drop duplicates while preserving order."""

    deduplicated: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for src, dst in edges:
        src_index = int(src)
        dst_index = int(dst)
        if not 0 <= src_index < num_nodes or not 0 <= dst_index < num_nodes:
            raise ValueError(
                f"Edge ({src_index}, {dst_index}) is out of range for graph with {num_nodes} nodes."
            )
        edge = (src_index, dst_index)
        if edge in seen:
            continue
        seen.add(edge)
        deduplicated.append(edge)
    return deduplicated


def get_num_nodes(layout: str) -> int:
    """Return the number of nodes for one reduced skeleton layout."""

    return len(GRAPH_NODE_NAMES[_validate_layout(layout)])


def get_node_names(layout: str) -> list[str]:
    """Return re-indexed node names for one reduced skeleton layout."""

    return list(GRAPH_NODE_NAMES[_validate_layout(layout)])


def get_inward_edges(layout: str) -> list[tuple[int, int]]:
    """Return directed inward edges for one layout."""

    layout_name = _validate_layout(layout)
    return _validate_edges(get_num_nodes(layout_name), GRAPH_INWARD_EDGES[layout_name])


def get_outward_edges(layout: str) -> list[tuple[int, int]]:
    """Return directed outward edges for one layout."""

    return [(dst, src) for src, dst in get_inward_edges(layout)]


def get_edges(layout: str, include_self: bool = False) -> list[tuple[int, int]]:
    """Return the canonical structural edge list for one layout."""

    layout_name = _validate_layout(layout)
    num_nodes = get_num_nodes(layout_name)
    edges = _validate_edges(num_nodes, GRAPH_EDGES[layout_name])
    if include_self:
        edges = edges + [(index, index) for index in range(num_nodes)]
    return edges


def build_adjacency_matrix(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    self_links: bool = True,
    undirected: bool = True,
) -> np.ndarray:
    """Build one dense adjacency matrix with optional self links and symmetry."""

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in _validate_edges(num_nodes, edges):
        adjacency[src, dst] = 1.0
        if undirected:
            adjacency[dst, src] = 1.0
    if self_links:
        adjacency += np.eye(num_nodes, dtype=np.float32)
    return adjacency


def normalize_adjacency(A: np.ndarray, method: str = "symmetric") -> np.ndarray:
    """Normalize an adjacency matrix safely even when isolated nodes exist."""

    adjacency = np.asarray(A, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got shape {adjacency.shape}.")
    if method != "symmetric":
        raise ValueError(f"Unsupported adjacency normalization method: {method!r}.")

    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float32)
    positive_mask = degree > 0.0
    inv_sqrt_degree[positive_mask] = np.power(degree[positive_mask], -0.5)
    diagonal = np.diag(inv_sqrt_degree)
    normalized = diagonal @ adjacency @ diagonal
    return normalized.astype(np.float32)


def build_stgcn_adjacency(
    layout: str,
    strategy: str = "spatial",
    *,
    add_self_links: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Build ST-GCN-style adjacency tensors for one reduced skeleton layout."""

    layout_name = _validate_layout(layout)
    num_nodes = get_num_nodes(layout_name)

    if strategy == "uniform":
        adjacency = build_adjacency_matrix(
            num_nodes,
            get_edges(layout_name, include_self=False),
            self_links=add_self_links,
            undirected=True,
        )
        if normalize:
            adjacency = normalize_adjacency(adjacency)
        return adjacency[None, ...].astype(np.float32)

    if strategy == "spatial":
        self_adjacency = build_adjacency_matrix(
            num_nodes,
            [],
            self_links=add_self_links,
            undirected=False,
        )
        inward_adjacency = build_adjacency_matrix(
            num_nodes,
            get_inward_edges(layout_name),
            self_links=False,
            undirected=False,
        )
        outward_adjacency = build_adjacency_matrix(
            num_nodes,
            get_outward_edges(layout_name),
            self_links=False,
            undirected=False,
        )
        if normalize:
            self_adjacency = normalize_adjacency(self_adjacency)
            inward_adjacency = normalize_adjacency(inward_adjacency)
            outward_adjacency = normalize_adjacency(outward_adjacency)
        return np.stack(
            [self_adjacency, inward_adjacency, outward_adjacency],
            axis=0,
        ).astype(np.float32)

    raise ValueError(
        f"Unsupported graph strategy: {strategy!r}. Expected 'uniform' or 'spatial'."
    )


class SkeletonGraph:
    """Container for one selected skeleton graph layout and adjacency tensor."""

    def __init__(
        self,
        layout: str = "selected_27",
        strategy: str = "spatial",
        *,
        normalize: bool = True,
        add_self_links: bool = True,
    ) -> None:
        self.layout = _validate_layout(layout)
        self.strategy = str(strategy)
        self.normalize = bool(normalize)
        self.add_self_links = bool(add_self_links)
        self.num_nodes = get_num_nodes(self.layout)
        self.node_names = get_node_names(self.layout)
        self.self_links = [(index, index) for index in range(self.num_nodes)] if self.add_self_links else []
        self.inward_edges = get_inward_edges(self.layout)
        self.outward_edges = get_outward_edges(self.layout)
        self.edges = get_edges(self.layout, include_self=False)
        self.A = build_stgcn_adjacency(
            self.layout,
            strategy=self.strategy,
            add_self_links=self.add_self_links,
            normalize=self.normalize,
        )

    def adjacency(self) -> np.ndarray:
        """Return the graph adjacency tensor."""

        return self.A

    def __repr__(self) -> str:
        return (
            "SkeletonGraph("
            f"layout={self.layout!r}, "
            f"strategy={self.strategy!r}, "
            f"num_nodes={self.num_nodes}, "
            f"num_edges={len(self.edges)}, "
            f"adjacency_shape={tuple(self.A.shape)})"
        )


__all__ = [
    "SkeletonGraph",
    "build_adjacency_matrix",
    "build_stgcn_adjacency",
    "get_edges",
    "get_inward_edges",
    "get_node_names",
    "get_num_nodes",
    "get_outward_edges",
    "normalize_adjacency",
]
