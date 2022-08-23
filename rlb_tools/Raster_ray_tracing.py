
import numpy as np
import random

from numba import jit

@jit
def bresenham(x0, y0, x1, y1):
    """
    Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def check_comms_available(pose_1: tuple, pose_2: tuple, obstacle_probabilities_grid: np.array) -> bool:
    pose_1 = list(pose_1)
    pose_2 = list(pose_2)

    pose_1.reverse()
    pose_2.reverse()

    # -> Adjusting coordinates
    pose_1[0] = obstacle_probabilities_grid.shape[0]- 1 - pose_1[0]
    pose_2[0] = obstacle_probabilities_grid.shape[0]- 1 - pose_2[0]

    # -> Fetching ray coordinates
    ray_coordinates = list(bresenham(
    x0=pose_1[0],
    y0=pose_1[1],
    x1=pose_2[0],
    y1=pose_2[1]))

    lst = []
    comms = True

    # -> Retreiving cooresponding values in comms obstacle grid
    for coordinate in ray_coordinates:
        lst.append(obstacle_probabilities_grid[coordinate])
        
        random_val = random.random()
        if random_val <= obstacle_probabilities_grid[coordinate]:
            comms = False
        else:
            pass
    
    # print(ray_coordinates)
    # print(lst)

    return comms, lst, ray_coordinates


if __name__ == "__main__":
    obstacle_grid = np.array((
        [10, 0, 0, 0, 0, 0, 0, 0, 11],
        [0, 1, 0, 0, 0, 0, 0, 15, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [12, 0, 0, 0, 0, 0, 0, 0, 13],
        [12, 0, 0, 0, 0, 0, 0, 0, 13]),
    )

    print(f"Comms availability: {check_comms_available((8, 9), (0, 0), obstacle_grid)}")
