class KMeans:
    def __init__(self, k):
        self.k = k # number of centres
        self.centres = []

    # Computing AVG.
    def point_avg(points):
        dimensions = len(points[0])
        new_center = []
        for dimension in xrange(dimensions):
            dim_sum = 0  # dimension sum
            for p in points:
                dim_sum += p[dimension]

            new_center.append(dim_sum / float(len(points)))
        return new_center

    # Assign Closest Center.
    def assign_points(self, samples):
        assignments = []
        for point in samples:
            shortest = distance(point, self.centres[i])
            shortest_index = 0
            for i in xrange(len(self.centers)):
                val = distance(point, self.centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            assignments.append(shortest_index)
        return assignments

    # From the Assignment Compute Centres.
    def update_centers(samples, assignments):
        new_means = defaultdict(list)
        self.centers = []
        for assignment, point in zip(assignments, samples):
            new_means[assignment].append(point)
        for points in new_means.itervalues():
            self.centers.append(point_avg(points))

    # Distance between two points.
    def distance(a,b):
        dimensions = len(a)
        _sum = 0
        for dimension in xrange(dimensions):
            difference_sq = (a[dimension] - b[dimension]) ** 2
            _sum += difference_sq
        return sqrt(_sum)

    # Compute Kmeans
    def compute(self, samples):
        self.centres = initCentres(samples, k)
        assignments = self.assign_points(samples, self.centres)
        old_assignments = None
        while assignments != old_assignments:
            update_centers(samples, assignments)
            old_assignments = assignments
            assignments = self.assign_points(samples)

    # Init Random Centres
    def initCentres(samples, k):
        self.centers = []
        dimensions = len(samples[0])
        min_max = defaultdict(int)

        for point in samples:
            for i in xrange(dimensions):
                val = point[i]
                min_key = 'min_%d' % i
                max_key = 'max_%d' % i
                if min_key not in min_max or val < min_max[min_key]:
                    min_max[min_key] = val
                if max_key not in min_max or val > min_max[max_key]:
                    min_max[max_key] = val

        for _k in xrange(k):
            rand_point = []
            for i in xrange(dimensions):
                min_val = min_max['min_%d' % i]
                max_val = min_max['max_%d' % i]

                rand_point.append(uniform(min_val, max_val))

            self.centers.append(rand_point)

    # Return Centres after running Kmeans
    def get_k_means(self):
        return self.centres
