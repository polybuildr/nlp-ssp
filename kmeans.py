from collections import defaultdict
import random

import numpy as np

from utils import distance

class KMeans:
    def __init__(self, k):
        self.k = k # number of centres
        self.centres = []

    # Computing AVG.
    def point_avg(self, points):
        dimensions = len(points[0])
        new_center = []
        for dimension in range(dimensions):
            dim_sum = 0  # dimension sum
            for p in points:
                dim_sum += p[dimension]

            new_center.append(dim_sum / float(len(points)))
        return new_center

    # Assign Closest Center.
    def assign_points(self, samples):
        assignments = []
        for point in samples:
            shortest = distance(point, self.centres[0])
            shortest_index = 0
            for i in range(len(self.centres)):
                val = distance(point, self.centres[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            assignments.append(shortest_index)
        return assignments

    # From the Assignment Compute Centres.
    def update_centres(self, samples, assignments):
        new_means = defaultdict(list)
        self.centres = []
        for assignment, point in zip(assignments, samples):
            new_means[assignment].append(point)
        for points in new_means.values():
            self.centres.append(self.point_avg(points))

    # Compute Kmeans
    def compute(self, samples):
        self.centres = self.initCentres(samples, self.k)
        assignments = self.assign_points(samples)
        old_assignments = None
        while assignments != old_assignments:
            self.update_centres(samples, assignments)
            old_assignments = assignments
            assignments = self.assign_points(samples)

    # Init Random Centres
    def initCentres(self, samples, k):
        self.centres = []
        dimensions = len(samples[0])
        min_max = defaultdict(int)

        for point in samples:
            for i in range(dimensions):
                val = point[i]
                min_key = 'min_%d' % i
                max_key = 'max_%d' % i
                if min_key not in min_max or val < min_max[min_key]:
                    min_max[min_key] = val
                if max_key not in min_max or val > min_max[max_key]:
                    min_max[max_key] = val

        for _k in range(k):
            rand_point = []
            for i in range(dimensions):
                min_val = min_max['min_%d' % i]
                max_val = min_max['max_%d' % i]

                rand_point.append(random.uniform(min_val, max_val))

            self.centres.append(rand_point)

    # Return Centres after running Kmeans
    def get_k_means(self):
        return self.centres
