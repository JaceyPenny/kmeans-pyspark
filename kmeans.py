import sys
import math
from typing import List, Tuple
from pyspark import SparkConf, SparkContext

# Define constants
DIMENSIONS = 20
THRESHOLD = 0.01
MAX_FLOAT_VALUE = sys.float_info.max


def parse_point(line: str) -> List[float]:
    point = [float(value) for value in line.split()]
    assert len(point) == DIMENSIONS, \
        'Invalid number of dimensions (must be {}). Input = "{}"'.format(DIMENSIONS, line)
    return point


def distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1) == DIMENSIONS, '"point1" must have dimensionality {}'.format(DIMENSIONS)
    assert len(point2) == DIMENSIONS, '"point2" must have dimensionality {}'.format(DIMENSIONS)

    inner_sum = 0

    for i in range(0, len(point1)):
        a = point1[i]
        b = point2[i]

        inner_sum += (a - b) ** 2

    return math.sqrt(inner_sum)


def clone_centroids(centroids_convert: List[List[float]]) -> List[List[float]]:
    clone = []
    for centroid in centroids_convert:
        centroid_clone = []
        for value in centroid:
            centroid_clone.append(value)
        clone.append(centroid_clone)

    return clone


def nearest_centroid(point: List[float]) -> Tuple[int, List[float]]:
    closest_centroid = None
    closest_distance = MAX_FLOAT_VALUE
    for i, centroid in enumerate(centroids):
        point_distance = distance(centroid, point)
        if point_distance < closest_distance:
            closest_distance = point_distance
            closest_centroid = i

    return closest_centroid, point


def point_add(point1: List[float], point2: List[float]):
    for i in range(len(point1)):
        point1[i] += point2[i]


def update_centroid(centroid: Tuple[int, List[List[float]]]) -> Tuple[int, List[float], float]:
    """
    Reduces a centroid (given it's ID and all the points belonging to it).
    The input is a tuple, where centroid[0] is the centroid's ID, and centroid[1]
    is the list of points belonging to the centroid.

    :param centroid:
    :return: A tuple of the centroid ID, the updated position of the centroid, and the cost metric
    """

    centroid_id = centroid[0]
    points = centroid[1]

    num_points = len(points)
    cost = 0

    new_centroid = [0 for _ in range(0, DIMENSIONS)]

    for point in points:
        cost += distance(centroids[centroid_id], point)
        point_add(new_centroid, point)

    return centroid_id, [value / num_points for value in new_centroid], cost


def centroid_diffs(centroids1: List[List[float]], centroids2: List[List[float]]) -> float:
    total_diff = 0.0
    for c1, c2 in zip(centroids1, centroids2):
        total_diff += distance(c1, c2)

    return total_diff


# ============ #
# MAIN PROGRAM #
# ============ #

# Initialize spark configuration and context
conf = SparkConf()
sc = SparkContext(conf=conf)

dataset = sc.textFile(sys.argv[1]).map(parse_point)
centroids = sc.textFile(sys.argv[2]).map(parse_point).collect()

updated_centroids = clone_centroids(centroids)

for c in centroids:
    print(c)

iterations = 0
costs = []

while True:
    # do stuff
    centroids = clone_centroids(updated_centroids)

    nearest_centroids = dataset.map(nearest_centroid)
    new_centroids = nearest_centroids.groupByKey().map(update_centroid).collect()

    total_cost = 0
    for index, centroid_i, cost_i in new_centroids:
        total_cost += cost_i
        updated_centroids[index] = centroid_i

    costs.append(total_cost)

    iterations += 1

    diff = centroid_diffs(centroids, updated_centroids)
    print(diff)
    if centroid_diffs(centroids, updated_centroids) < THRESHOLD:
        break

# print centroids
print('\n\nCOMPLETED IN {} ITERATIONS\n\nCENTROID RESULTS\n========================'.format(iterations))
for c in centroids:
    print(c)

print('\n\n')
print('COSTS\n==========================')
print('{:>12}  {:>12}'.format('iteration', 'cost'))
for i, cost_i in enumerate(costs):
    print('{:>12}  {:>12}'.format(i, round(cost_i, 4)))
print('\n\n')
