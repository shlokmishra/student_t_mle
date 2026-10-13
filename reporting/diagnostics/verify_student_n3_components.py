"""Exact chamber graph and low-dimensional MLE check for the n=3 score manifold."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reporting.diagnostics.study_student_mle_transitions import mle_competitor_margin
from reporting.diagnostics.study_student_n3_branch_components import x_from_z_branch


Chamber = tuple[tuple[int, int, int], tuple[int, int, int]]


def chambers() -> list[Chamber]:
    """Return all nonempty finite chambers; zero denotes an inner-coordinate sign."""
    result = []
    for bits in itertools.product((0, 1), repeat=3):
        outer = [index for index, bit in enumerate(bits) if bit]
        for assigned in itertools.product((-1, 1), repeat=len(outer)):
            signs = [0, 0, 0]
            for index, sign in zip(outer, assigned):
                signs[index] = sign
            # Three score coordinates cannot all have the same strict sign.
            if len(outer) == 3 and abs(sum(signs)) == 3:
                continue
            result.append((bits, tuple(signs)))
    return result


def adjacent(left: Chamber, right: Chamber) -> bool:
    """Whether closures meet by one finite inner/outer merge."""
    left_bits, left_signs = left
    right_bits, right_signs = right
    changed = [i for i in range(3) if left_bits[i] != right_bits[i]]
    if len(changed) != 1:
        return False
    coordinate = changed[0]
    for i in range(3):
        if i != coordinate and (left_bits[i] != right_bits[i] or left_signs[i] != right_signs[i]):
            return False
    outer = right if right_bits[coordinate] else left
    merge_sign = outer[1][coordinate]
    other_outer_signs = [
        outer[1][i] for i in range(3) if i != coordinate and outer[0][i]
    ]
    return all(sign == -merge_sign for sign in other_outer_signs)


def chamber_graph() -> dict[Chamber, set[Chamber]]:
    nodes = chambers()
    graph = {node: set() for node in nodes}
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if adjacent(left, right):
                graph[left].add(right)
                graph[right].add(left)
    return graph


def connected_components(graph: dict[Chamber, set[Chamber]]) -> list[set[Chamber]]:
    unseen = set(graph)
    result = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return sorted(result, key=lambda component: (-len(component), sorted(component)))


def predicted_mle(chamber: Chamber) -> bool:
    outer_signs = [sign for bit, sign in zip(*chamber) if bit]
    return not (outer_signs.count(1) >= 2 or outer_signs.count(-1) >= 2)


def chamber_name(chamber: Chamber) -> str:
    bits, signs = chamber
    code = "".join(map(str, bits))
    sign_code = "".join("+" if sign > 0 else "-" if sign < 0 else "." for sign in signs)
    return f"{code}:{sign_code}"


def verify_mle_grid(resolution: int) -> dict:
    values = np.linspace(-1.0, 1.0, resolution)
    checked = mismatches = 0
    by_chamber: dict[str, int] = {}
    for chamber in chambers():
        bits, signs = chamber
        count = 0
        for u1 in values:
            for u2 in values:
                u = np.asarray([u1, u2, -u1 - u2])
                if np.max(np.abs(u)) >= 1.0 - 1e-12:
                    continue
                if any(bit and (abs(value) < 1e-10 or int(np.sign(value)) != sign)
                       for bit, sign, value in zip(bits, signs, u)):
                    continue
                x = x_from_z_branch(0.5 * u, bits, 1.0, 0.0)
                actual = mle_competitor_margin(x, 0.0, 1.0)["mu_star_is_global_mle"]
                checked += 1
                count += 1
                mismatches += int(actual != predicted_mle(chamber))
        by_chamber[chamber_name(chamber)] = count
    return {"checked": checked, "mismatches": mismatches, "points_by_chamber": by_chamber}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=41)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph = chamber_graph()
    components = connected_components(graph)
    report = {
        "num_chambers": len(graph),
        "num_edges": sum(map(len, graph.values())) // 2,
        "num_components": len(components),
        "components": [
            {
                "size": len(component),
                "predicted_mle": all(predicted_mle(node) for node in component),
                "chambers": [chamber_name(node) for node in sorted(component)],
            }
            for component in components
        ],
        "grid_verification": verify_mle_grid(args.resolution),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
