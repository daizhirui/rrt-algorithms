# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

from rrt_algorithms.utilities.geometry import es_points_along_line
from rrt_algorithms.utilities.obstacle_generation import obstacle_generator


class SearchSpace(object):
    def __init__(self, dimension_lengths, obstacle_free, collision_free):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param obstacle_free: function to check if a location is inside an obstacle
        :param collision_free: function to check if a line segment intersects an obstacle
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        self.obstacle_free = obstacle_free
        self.collision_free = collision_free

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside an obstacle
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)


class ObstacleRtree:
    def __init__(self, dim: int, obstacles):
        p = index.Property()
        p.dimension = dim

        if obstacles is None:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            if any(len(o) / 2 != dim for o in obstacles):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in obstacles for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            self.obs = index.Index(obstacle_generator(obstacles), interleaved=True, properties=p)

    def obstacle_free(self, x):
        """
        Check if a location resides inside an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        return self.obs.count(x) == 0

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        points = es_points_along_line(start, end, r)
        coll_free = all(map(self.obstacle_free, points))
        return coll_free
