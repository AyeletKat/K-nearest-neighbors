
from collections import Counter
import random
import numpy as np

# implementation of knn algorithm using iris dataset

file = open("iris.txt", "r")
data = file.readlines()
file.close()
# for i in range (5):   # printout for check 
#     print(data[i].strip())

# Splitting the data into 3 parts - setosa, versicolor, virginica
setosa = []
versicolor = []
virginica = []

for i in range (len(data)):
    if "setosa" in data[i]:
        setosa.append(data[i].strip())
    elif "versicolor" in data[i]:
        versicolor.append(data[i].strip())
    elif "virginica" in data[i]:
        virginica.append(data[i].strip())
# print("\nSetosa samples:")    # printout for check 
# for i in range (3):
#     print(setosa[i])
# print("\nVersicolor samples:")
# for i in range (3):
#     print(versicolor[i])
# print("\nVirginica samples:")
# for i in range (3):
#     print(virginica[i])
# stripping the arrays to keep onlt the second and third parameters as asked in the assignment
setosa = [i.split(" ")[1:3] for i in setosa]
versicolor = [i.split(" ")[1:3] for i in versicolor]
virginica = [i.split(" ")[1:3] for i in virginica]

# converting the arrays to float
setosa = [[float(i[0]), float(i[1])] for i in setosa]
versicolor = [[float(i[0]), float(i[1])] for i in versicolor]
virginica = [[float(i[0]), float(i[1])] for i in virginica]

# labeling as 0s and 1s
versi_virgi = versicolor + virginica
all_labels1 = [0]*len(versicolor) + [1]*len(virginica)

setosa_virgi = setosa + virginica
all_labels2 = [0]*len(setosa) + [1]*len(virginica)

# # printing the first 3 samples of each class  # printout for check 
# print("\nSetosa samples after stripping and converting to float:")
# for i in range (3):
#     print(setosa[i])
# print("\nVersicolor samples after stripping and converting to float:")
# for i in range (3):
#     print(versicolor[i])
# print("\nVirginica samples after stripping and converting to float:")
# for i in range (3):
#     print(virginica[i])

# #  create a scatter plot of the data   # printout for check 
# import matplotlib.pyplot as plt
# plt.scatter([i[0] for i in setosa], [i[1] for i in setosa], color='red', label='Setosa')
# plt.scatter([i[0] for i in versicolor], [i[1] for i in versicolor], color='blue', label='Versicolor')
# plt.scatter([i[0] for i in virginica], [i[1] for i in virginica], color='green', label='Virginica')
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.title('Iris Dataset')
# plt.legend()
# plt.show()

def l1_distance(point1, point2):
    """Calculate the L1 distance between two points."""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def l2_distance(point1, point2):
    dis0 = abs(point1[0]-point2[0]) ** 2
    dis1 = abs(point1[1]-point2[1]) ** 2
    return (dis0 + dis1) ** 0.5

def l_inf_distance(point1, point2):
    """Calculate the L-infinity distance between two points."""
    return max(abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))


# the knn prediction function, checks distances based on current distance metric,
# and classifes the point according to most common label among the k nearest neighbors
def knn_predict(training_data, training_labels, test_point, k, distance_fn):
    distances = []
    for i in range(len(training_data)):
        dist = distance_fn(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    return Counter(k_nearest).most_common(1)[0][0]


# sends organized data to the knn function, creates the experiments according to k (num neighbors) and p (what distance metric to use)
# returns the average error for each combination of k and p 
def run_experiment(k_list, p_list, all_data, all_labels, repetitions=100):
    errors = { (k, p): {'train': [], 'test': []} for k in k_list for p in p_list }

    distance_map = {
        1: l1_distance,
        2: l2_distance,
        float('inf'): l_inf_distance
    }

    for _ in range(repetitions):
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        split = len(indices) // 2
        train_idx = indices[:split]
        test_idx = indices[split:]

        train_data = [all_data[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]

        for k in k_list:
            for p in p_list:
                dist_fn = distance_map[p]

                train_preds = [knn_predict(train_data, train_labels, x, k, dist_fn) for x in train_data]
                test_preds = [knn_predict(train_data, train_labels, x, k, dist_fn) for x in test_data]

                train_err = sum([pred != true for pred, true in zip(train_preds, train_labels)]) / len(train_labels)
                test_err = sum([pred != true for pred, true in zip(test_preds, test_labels)]) / len(test_labels)

                errors[(k, p)]['train'].append(train_err)
                errors[(k, p)]['test'].append(test_err)

    # --- Averaging results ---
    results = {}
    for kp in errors:
        train_mean = np.mean(errors[kp]['train'])
        test_mean = np.mean(errors[kp]['test'])
        gap = abs(train_mean - test_mean)
        results[kp] = (train_mean, test_mean, gap)

    return results

#  versi_virgi results, regular knn
k_values = [1, 3, 5, 7, 9]
p_values = [1, 2, float('inf')]
results = run_experiment(k_values, p_values, versi_virgi, all_labels1, repetitions=100)

for (k, p), (train_err, test_err, diff) in sorted(results.items()):
    print(f"k={k}, p={p}: train error={train_err:.3f}, test error={test_err:.3f}, |gap|={diff:.3f}")


# ################################# PART D , condensed, epsilon net.

# checks distances according to recieved epsilon value and distance function,
# builds a condensed set of data, where each point is at least epsilon away from all other points in the set
# and all the rest of the points are closer then epsilon to a point in the set

def build_condensed_net(data, labels, epsilon, distance_fn):
    T_data = []
    T_labels = []

    for i in range(len(data)):
        p = data[i]
        if not T_data:
            T_data.append(p)
            T_labels.append(labels[i])
            continue

        min_dist = min(distance_fn(p, q) for q in T_data)
        if min_dist > epsilon:
            T_data.append(p)
            T_labels.append(labels[i])

    return T_data, T_labels

def find_min_margin(class0, class1, distance_fn):
    min_margin = float('inf')
    for p in class0:
        for q in class1:
            d = distance_fn(p, q)
            if d < min_margin:
                min_margin = d
    # print(f"Minimum margin found: {min_margin:.3f}") # checking
    return min_margin

# basically same as run_experiment, but uses the condensed set of data
# and calculates the margin between the two classes in the training set to determine epsilon (calls find_min_margin)
# returns errors averages list.
def run_experiment_condensed(k_list, p_list, all_data, all_labels, repetitions=100):
    errors = { (k, p): {'train': [], 'test': [], 'condensed_sizes': []} for k in k_list for p in p_list }

    distance_map = {
        1: l1_distance,
        2: l2_distance,
        float('inf'): l_inf_distance
    }

    for _ in range(repetitions):
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        split = len(indices) // 2
        train_idx = indices[:split]
        test_idx = indices[split:]

        train_data = [all_data[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]

        for k in k_list:
            for p in p_list:
                dist_fn = distance_map[p]
                # Determine ε as the margin between the two classes in the training set
                class0_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 0]
                class1_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 1]
                epsilon = find_min_margin(class0_data, class1_data, dist_fn)
                # print(f"Using ε={epsilon:.3f} for k={k}, p={p} in repetition {_+1}/{repetitions}")
                # Build condensed training set using ε-net
                condensed_data, condensed_labels = build_condensed_net(train_data, train_labels, epsilon, dist_fn)

                # Leave-one-out error on the condensed set:
                train_preds = [
                    knn_predict(
                        [condensed_data[j] for j in range(len(condensed_data)) if j != i],
                        [condensed_labels[j] for j in range(len(condensed_labels)) if j != i],
                        condensed_data[i],
                        k,
                        dist_fn
                    )
                    for i in range(len(condensed_data))
                ]

                test_preds = [knn_predict(condensed_data, condensed_labels, x, k, dist_fn) for x in test_data]

                train_err = sum([pred != true for pred, true in zip(train_preds, condensed_labels)]) / len(condensed_labels)
                test_err = sum([pred != true for pred, true in zip(test_preds, test_labels)]) / len(test_labels)

                errors[(k, p)]['train'].append(train_err)
                errors[(k, p)]['test'].append(test_err)
                errors[(k, p)]['condensed_sizes'].append(len(condensed_data))

    results = {}
    for kp in errors:
        train_mean = np.mean(errors[kp]['train'])
        test_mean = np.mean(errors[kp]['test'])
        size_mean = np.mean(errors[kp]['condensed_sizes'])
        gap = abs(train_mean - test_mean)
        results[kp] = (train_mean, test_mean, gap, size_mean)

    return results

k_values = [1, 3, 5, 7, 9]
p_values = [1, 2, float('inf')]
# versi_virgi condesed results
condensed_results = run_experiment_condensed(k_values, p_values, versi_virgi, all_labels1, repetitions=100)
for (k, p), (train_err, test_err, diff, size) in sorted(condensed_results.items()):
    p_str = "inf" if p == float('inf') else str(p)
    print(f"k={k}, p={p_str}: train error={train_err:.3f}, test error={test_err:.3f}, |gap|={diff:.3f}, avg condensed size={size:.1f}")


################################# PART E - running again but on setosa & virginica 
# setosa_virgi regular results
results = run_experiment(k_values, p_values, versi_virgi, all_labels1, repetitions=100)

for (k, p), (train_err, test_err, diff) in sorted(results.items()):
    print(f"k={k}, p={p}: train error={train_err:.3f}, test error={test_err:.3f}, |gap|={diff:.3f}")

# setosa_virgi condesed results
condensed_results = run_experiment_condensed(k_values, p_values, setosa_virgi, all_labels2, repetitions=100)
for (k, p), (train_err, test_err, diff, size) in sorted(condensed_results.items()):
    p_str = "inf" if p == float('inf') else str(p)
    print(f"k={k}, p={p_str}: train error={train_err:.3f}, test error={test_err:.3f}, |gap|={diff:.3f}, avg condensed size={size:.1f}  | setosa&virginica")
