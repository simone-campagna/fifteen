import collections
import itertools
import math
import queue


__all__ = [
    'a_star',
    'ida_star',
    'get_algorithms',
    'get_algorithm',
]


def identity(obj):
    return obj


def reconstruct_path(parent, current, get_idx=identity):
    total_path = [current]
    current_idx = get_idx(current)
    while current_idx in parent:
        current = parent[current_idx]
        current_idx = get_idx(current)
        total_path.append(current)
    total_path.reverse()
    return total_path


# REM def a_star_base(start, found, *, get_neighbors, heuristic_cost, get_idx=identity, tracker=None):
# REM     # function A_Star(start, goal)
# REM     #     // The set of nodes already evaluated
# REM     #     closedSet := {}
# REM     closed_set = set()
# REM     # 
# REM     #     // The set of currently discovered nodes that are not evaluated yet.
# REM     #     // Initially, only the start node is known.
# REM     #     openSet := {start}
# REM     open_set = {start}
# REM     # 
# REM     #     // For each node, which node it can most efficiently be reached from.
# REM     #     // If a node can be reached from many nodes, cameFrom will eventually contain the
# REM     #     // most efficient previous step.
# REM     #     cameFrom := an empty map
# REM     came_from = {}
# REM     # 
# REM     #     // For each node, the cost of getting from the start node to that node.
# REM     #     gScore := map with default value of Infinity
# REM     inf = math.inf
# REM     g_score = collections.defaultdict(lambda: inf)
# REM     # 
# REM     #     // The cost of going from start to start is zero.
# REM     #     gScore[start] := 0
# REM     g_score[start] = 0
# REM     # 
# REM     #     // For each node, the total cost of getting from the start node to the goal
# REM     #     // by passing by that node. That value is partly known, partly heuristic.
# REM     #     fScore := map with default value of Infinity
# REM     f_score = collections.defaultdict(lambda: inf)
# REM     # 
# REM     #     // For the first node, that value is completely heuristic.
# REM     #     fScore[start] := heuristic_cost_estimate(start, goal)
# REM     f_score[start] = heuristic_cost(start)
# REM     # 
# REM     #     while openSet is not empty
# REM     while open_set:
# REM     #         current := the node in openSet having the lowest fScore[] value
# REM     #         if current = goal
# REM     #             return reconstruct_path(cameFrom, current)
# REM         current = sorted(open_set, key=lambda x: f_score[x])[0]
# REM         if found(current):
# REM             return reconstruct_path(came_from, current)
# REM     # 
# REM     #         openSet.Remove(current)
# REM     #         closedSet.Add(current)
# REM         open_set.remove(current)
# REM         closed_set.add(current)
# REM     # 
# REM     #         for each neighbor of current
# REM     #             if neighbor in closedSet
# REM     #                 continue		// Ignore the neighbor which is already evaluated.
# REM         for neighbor, move, distance in get_neighbors(current):
# REM             if neighbor in closed_set:
# REM                 continue
# REM     # 
# REM     #             // The distance from start to a neighbor
# REM     #             tentative_gScore := gScore[current] + dist_between(current, neighbor)
# REM             tentative_g_score = g_score[current] + distance
# REM     # 
# REM     #             if neighbor not in openSet	// Discover a new node
# REM     #                 openSet.Add(neighbor)
# REM     #             else if tentative_gScore >= gScore[neighbor]
# REM     #                 continue		// This is not a better path.
# REM             if neighbor not in open_set:
# REM                 open_set.add(neighbor)
# REM             elif tentative_g_score >= g_score[neighbor]:
# REM                 continue
# REM     # 
# REM     #             // This path is the best until now. Record it!
# REM     #             cameFrom[neighbor] := current
# REM     #             gScore[neighbor] := tentative_gScore
# REM     #             fScore[neighbor] := gScore[neighbor] + heuristic_cost_estimate(neighbor, goal)
# REM             came_from[neighbor] = current
# REM             g_score[neighbor] = tentative_g_score
# REM             f_score[neighbor] = tentative_g_score + heuristic_cost(neighbor)
# REM             if tracker:
# REM                 tracker.add(1)


def a_star(start, found, *, get_neighbors, heuristic_cost, get_idx=identity, tracker=None):
    ith = itertools.count()
    came_from = {}
    inf = math.inf
    start_idx = get_idx(start)
    g_score = collections.defaultdict(lambda: inf)
    g_score[start_idx] = 0
    f_score = collections.defaultdict(lambda: inf)
    f_score[start_idx] = heuristic_cost(start)
    closed_set = set()
    open_set = {get_idx(start)}
    open_set_queue = queue.PriorityQueue()
    open_set_queue.put_nowait((f_score[start_idx], next(ith), start))
    while open_set_queue:
        current = open_set_queue.get_nowait()[-1]
        current_idx = get_idx(current)
        if found(current):
            return reconstruct_path(came_from, current, get_idx=get_idx)

        closed_set.add(current_idx)

        for neighbor, move, distance in get_neighbors(current):
            neighbor_idx = get_idx(neighbor)
            if neighbor_idx in closed_set:
                continue

            tentative_g_score = g_score[current_idx] + distance
            f_score_neighbor = tentative_g_score + heuristic_cost(neighbor)
            if tracker:
                tracker.add(1)

            neighbor_idx = get_idx(neighbor)
            if neighbor_idx not in open_set:
                open_set_queue.put_nowait((f_score_neighbor, next(ith), neighbor))
                open_set.add(neighbor_idx)
            elif tentative_g_score >= g_score[neighbor_idx]:
                continue

            came_from[neighbor_idx] = current
            g_score[neighbor_idx] = tentative_g_score
            f_score[neighbor_idx] = f_score_neighbor


class IdaStarFound(Exception):
    pass


def ida_star(start, found, *, get_neighbors, heuristic_cost, get_idx=identity, tracker=None):
    inf = math.inf
    def search(path, s_path, g, bound, found, get_neighbors, heuristic_cost, tracker):
        node = path[-1]
        f = g + heuristic_cost(node)
        if tracker:
            tracker.add(1)
        if f > bound:
            return f
        if found(node):
            raise IdaStarFound()
        minimum = inf
        for neighbor, move, distance in get_neighbors(node):
            neighbor_idx = get_idx(neighbor)
            if neighbor_idx not in s_path:
                path.append(neighbor)
                s_path.add(neighbor_idx)
                t = search(path, s_path, g + distance, bound, found, get_neighbors, heuristic_cost, tracker)
                if t < minimum:
                    minimum = t
                s_path.discard(get_idx(path.pop(-1)))
        return minimum

    bound = heuristic_cost(start)
    path = [start]
    start_idx = get_idx(start)
    s_path = {start_idx}
    try:
        while True:
            bound = search(path, s_path, 0, bound, found, get_neighbors, heuristic_cost, tracker)
    except IdaStarFound as isf:
        return path


ALGORITHMS = collections.OrderedDict([(a.__name__, a) for a in (a_star, ida_star)])

def get_algorithms():
    yield from ALGORITHMS


def get_algorithm(algorithm):
    hfunction = ALGORITHMS.get(algorithm, None)
    if hfunction is None:
        raise ValueError("unknown algorithm {}".format(algorithm))
    return hfunction

