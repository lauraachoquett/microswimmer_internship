import heapq
import math
def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(start, goal, grid, min_distance):
    start = tuple(start)
    goal = tuple(goal)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            if is_valid_move(neighbor, grid, min_distance):
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def get_neighbors(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def is_valid_move(point, grid, min_distance):
    x, y = point
    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
        return False
    if grid[x, y] < min_distance:
        return False
    return True
