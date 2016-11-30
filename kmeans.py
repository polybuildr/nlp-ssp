from collections import defaultdict
import random

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as SKMeans

from utils import distance

class KMeans:
    def __init__(self, k):
        self.k = k # number of centres
        self.kmeans = MiniBatchKMeans(n_clusters=k, init='random')
        # self.kmeans = SKMeans(n_clusters=k, init='random')
        self.centres = []

    def compute(self, samples):
        # self.kmeans.fit(samples)
        n = 10000
        for batch in [samples[i:i + n] for i in range(0, len(samples), n)]:
            self.kmeans.partial_fit(batch)
            print("Fit one batch...")
        self.centres = self.kmeans.cluster_centers_

    def get_k_means(self):
        return self.kmeans.cluster_centers_

    def assign_points(self, samples):
        return self.kmeans.predict(samples)

    # # Computing AVG.
    # def point_avg(self, points):
    #     dimensions = len(points[0])
    #     new_center = []
    #     for dimension in range(dimensions):
    #         dim_sum = 0  # dimension sum
    #         for p in points:
    #             dim_sum += p[dimension]
    #
    #         new_center.append(dim_sum / float(len(points)))
    #     return new_center
    #
    # Assign Closest Center.
    # def assign_points(self, samples):
    #     assignments = []
    #     for point in samples:
    #         shortest = distance(point, self.centres[0])
    #         shortest_index = 0
    #         for i in range(len(self.centres)):
    #             val = distance(point, self.centres[i])
    #             if val < shortest:
    #                 shortest = val
    #                 shortest_index = i
    #         assignments.append(shortest_index)
    #     return assignments

    # # From the Assignment Compute Centres.
    # def update_centres(self, samples, assignments):
    #     new_means = defaultdict(list)
    #     self.centres = []
    #     for assignment, point in zip(assignments, samples):
    #         new_means[assignment].append(point)
    #     for points in new_means.values():
    #         self.centres.append(self.point_avg(points))
    #
    # # Compute Kmeans
    # def compute(self, samples):
    #     # print('centres, ', self.centres)
    #     self.initCentres(samples, self.k)
    #     # print('centres, ', self.centres)
    #     assignments = self.assign_points(samples)
    #     # print('centres, ', self.centres)
    #     old_assignments = None
    #     iter_count = 0
    #     while assignments != old_assignments:
    #         self.update_centres(samples, assignments)
    #         old_assignments = assignments
    #         assignments = self.assign_points(samples)
    #         iter_count += 1
    #         print(iter_count)
    #         if iter_count > 1:
    #             break
    #
    # # Init Random Centres
    # def initCentres(self, samples, k):
    #     # print('centres, ', self.centres)
    #     self.centres = []
    #     # print('centres, ', self.centres)
    #     dimensions = len(samples[0])
    #     min_max = defaultdict(int)
    #
    #     for point in samples:
    #         for i in range(dimensions):
    #             val = point[i]
    #             min_key = 'min_%d' % i
    #             max_key = 'max_%d' % i
    #             if min_key not in min_max or val < min_max[min_key]:
    #                 min_max[min_key] = val
    #             if max_key not in min_max or val > min_max[max_key]:
    #                 min_max[max_key] = val
    #
    #     for _k in range(k):
    #         rand_point = []
    #         for i in range(dimensions):
    #             min_val = min_max['min_%d' % i]
    #             max_val = min_max['max_%d' % i]
    #
    #             rand_point.append(random.uniform(min_val, max_val))
    #
    #         self.centres.append(rand_point)
    #     # print('centres, ', self.centres)
    #
    # # Return Centres after running Kmeans
    # def get_k_means(self):
    #     return self.centres
