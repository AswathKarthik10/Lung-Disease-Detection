'''
import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14], dtype = int)
y = np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28], dtype = int)

model.fit(x, y, epochs = 500)

print(model.predict([127180]))
'''
'''
import tensorflow as tf
from tensorflow import keras

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy')

model.fit(train_images, train_labels, 10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_loss, test_acc)
'''
'''
def calculate_distance(path, distances):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distances[path[i]][path[i + 1]]
    total_distance += distances[path[-1]][path[0]]
    return total_distance

def dfs_tsp(current_city, cities, distances, visited, path, depth_limit):
    visited[current_city] = True
    path.append(current_city)
    if len(path) == len(cities) or depth_limit == 0:
        return path
    for next_city in range(len(cities)):
        if not visited[next_city]:
            result = dfs_tsp(next_city, cities, distances, visited.copy(), path.copy(), depth_limit)
            if result:
                return result
    return None

def bfs_tsp(start_city, cities, distances):
    queue = [(start_city, [start_city])]
    while queue:
        current_city, path = queue.pop(0)
        for next_city in range(len(cities)):
            if next_city not in path:
                new_path = path + [next_city]
                if len(new_path) == len(cities):
                    return new_path
                queue.append((next_city, new_path))
    return None

def ids_tsp(start_city, cities, distances, depth_limit):
    for d in range(1, depth_limit + 1):
        visited = [False] * len(cities)
        result = dfs_tsp(start_city, cities, distances, visited, [], d)
        if result:
            return result
    return None

def tsp_solver(cities, distances):
    start_city = 0
    max_depth = len(cities)
    # DFS
    dfs_solution = dfs_tsp(start_city, cities, distances, [False] * len(cities), [], max_depth)
    dfs_distance = calculate_distance(dfs_solution, distances)
    # BFS
    bfs_solution = bfs_tsp(start_city, cities, distances)
    bfs_distance = calculate_distance(bfs_solution, distances)
    # IDS
    ids_solution = None
    ids_distance = None
    for depth_limit in range(1, max_depth + 1):
        ids_solution = ids_tsp(start_city, cities, distances, depth_limit)
        if ids_solution:
            ids_distance = calculate_distance(ids_solution, distances)
            break
    return {
        "DFS Solution": dfs_solution,
        "DFS Distance": dfs_distance,
        "BFS Solution": bfs_solution,
        "BFS Distance": bfs_distance,
        "IDS Solution": ids_solution,
        "IDS Distance": ids_distance
    }

# Example Usage
cities = [0, 1, 2, 3]
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

result = tsp_solver(cities, distances)
print("DFS Solution:", result["DFS Solution"], "Distance:", result["DFS Distance"])
print("BFS Solution:", result["BFS Solution"], "Distance:", result["BFS Distance"])
print("IDS Solution:", result["IDS Solution"], "Distance:", result["IDS Distance"])

'''
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def brute_force_closest_pair(points):
    min_distance = float('inf')
    closest_pair = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = euclidean_distance(points[i], points[j])
            if distance < min_distance:
                min_distance = distance
                closest_pair = (points[i], points[j])
    return closest_pair, min_distance

def closest_pair_recursive(points):
    n = len(points)
    if n <= 3:
        return brute_force_closest_pair(points)

    sorted_points = sorted(points, key=lambda x: x[0])
    mid = n // 2
    left_half = sorted_points[:mid]
    right_half = sorted_points[mid:]

    left_pair, left_distance = closest_pair_recursive(left_half)
    right_pair, right_distance = closest_pair_recursive(right_half)

    min_distance = min(left_distance, right_distance)
    min_pair = left_pair if left_distance <= right_distance else right_pair

    strip = []
    for point in sorted_points:
        if abs(point[0] - sorted_points[mid][0]) < min_distance:
            strip.append(point)

    strip.sort(key=lambda x: x[1])

    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < min_distance:
            distance = euclidean_distance(strip[i], strip[j])
            if distance < min_distance:
                min_distance = distance
                min_pair = (strip[i], strip[j])
            j += 1

    return min_pair, min_distance

def closest_pair(points):
    if len(points) < 2:
        return None, float('inf')
    return closest_pair_recursive(points)

# Example Usage:
points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
closest_pair, distance = closest_pair(points)
print("Closest Pair:", closest_pair, "Distance:", distance)
