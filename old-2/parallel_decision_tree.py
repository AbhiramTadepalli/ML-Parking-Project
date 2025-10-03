# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

# AXT220137 - Abhiram Tadepalli
# AXS220399 - Anveetha Suresh
# CS4375.001

import numpy as np
import pandas as pd
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def find_best_split_parallel(x, y, attribute_indices):
    """Parallel implementation to find the best feature to split on"""
    
    # Calculate information gain for each feature in parallel
    results = Parallel(n_jobs=-1)(
        delayed(calculate_feature_gain)(x, y, attr_idx) 
        for attr_idx in attribute_indices
    )
    
    # Find the best feature based on information gain
    best_idx = np.argmax([gain for attr_idx, gain in results])
    return results[best_idx]

def calculate_feature_gain(x, y, attr_idx):
    """Calculate information gain for a single feature"""
    # Your existing information gain calculation
    gain = mutual_information(x[:, attr_idx], y)
    return attr_idx, gain



def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
  # Get unique values and their indices
    unique_values, inverse_indices = np.unique(x, return_inverse=True)
    
    # Create a dictionary to store the partitions
    partitions = {}
    
    # Iterate through unique values and find their indices
    for i, value in enumerate(unique_values):
        partitions[value] = np.where(inverse_indices == i)[0]
    
    return partitions
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    # # INSERT YOUR CODE HERE
    # # count the frequency of zeros and ones
    # count_zeros = (y == 0).sum()
    # count_ones = len(y) - count_zeros
    # prob_zero = count_zeros / len(y)
    # prob_one = count_ones / len(y)

    # # calculate entropy
    # entropy = 0
    # if prob_zero > 0: # so the log doesn't have an error
    #     entropy -= prob_zero * np.log2(prob_zero) # negative sum so entropy is positive
    # if prob_one > 0:
    #     entropy -= prob_one * np.log2(prob_one)
    # return entropy

    # Vectorized implementation
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    entropy_y = entropy(y)
    # # H(y | X) = sum over x of [P(X = x) * H(y | X=x)]
    # cond_entropy_y_given_X = 0
    # for xi in np.unique(x): # iterate over each unique value of attribute x
    #     # P(X = x)
    #     p_xi = (x == xi).sum() / len(x)
    #     # H(y | X=x)
    #     yi = y[(x == xi)] # use a boolean mask so that yi contains only labels corrsponding to the current xi
    #     cond_entropy_y_given_X += p_xi * entropy(yi) # add weighted entropy for each unique value

    # return entropy_y - cond_entropy_y_given_X # mutual information
    # Get unique values and their counts in one operation
    values, counts = np.unique(x, return_counts=True)
    weighted_entropy = 0
    
    # Pre-compute indices for each value to avoid repeated boolean masking
    partitions = partition(x)
    
    for value, count in zip(values, counts):
        p_value = count / len(x)
        indices = partitions[value]
        weighted_entropy += p_value * entropy(y[indices])
        
    return entropy_y - weighted_entropy
    raise Exception('Function not yet implemented!')
    
def parallel_id3(x, y, attribute_indices=None, depth=0, max_depth=5, min_samples=2):
    # Initialize attribute indices on first call
    if attribute_indices is None:
        attribute_indices = list(range(x.shape[1]))
    
    # Base cases remain the same
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    if len(attribute_indices) == 0 or depth >= max_depth or len(y) < min_samples:
        return np.bincount(y.astype(int)).argmax()
    
    # Find best feature to split on
    best_attr, best_gain = find_best_split_parallel(x, y, attribute_indices)
    
    # If no information gain, return majority class
    if best_gain <= 0:
        return np.bincount(y.astype(int)).argmax()
    
    # Create tree node
    tree = {}
    remaining_attributes = [attr for attr in attribute_indices if attr != best_attr]
    
    # Get unique values for best attribute
    unique_values = np.unique(x[:, best_attr])
    
    # Process each branch in parallel if depth is small enough
    # (to avoid thread explosion at deeper levels)
    if depth < 3:  # Limit parallelism to upper levels of the tree
        results = Parallel(n_jobs=-1)(
            delayed(process_branch)(x, y, best_attr, value, remaining_attributes, depth, max_depth, min_samples)
            for value in unique_values
        )
        
        # Build tree from parallel results
        for value, subtree in zip(unique_values, results):
            tree[(best_attr, value)] = subtree
    else:
        # Sequential processing for deeper levels
        for value in unique_values:
            indices = x[:, best_attr] == value
            if np.sum(indices) == 0:
                tree[(best_attr, value)] = np.bincount(y.astype(int)).argmax()
            else:
                tree[(best_attr, value)] = parallel_id3(
                    x[indices], y[indices], remaining_attributes, 
                    depth + 1, max_depth, min_samples
                )
    
    return tree

def process_branch(x, y, attr_idx, value, remaining_attributes, depth, max_depth, min_samples):
    """Process a single branch of the decision tree"""
    indices = x[:, attr_idx] == value
    if np.sum(indices) == 0:
        return np.bincount(y.astype(int)).argmax()
    else:
        return parallel_id3(
            x[indices], y[indices], remaining_attributes, 
            depth + 1, max_depth, min_samples
        )



def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if isinstance(tree, dict) == False:
        return tree  # return label

    # Iterate through  items in the tree
    for criterion, subtree in tree.items():
        index, value, result = criterion

        if (x[index] == value) == result:   # Check if the value of the example matches the nodes value

            # if the subtree is a full, recursively descend the tree
            if isinstance(subtree, dict):
                return predict_example(x, subtree)
            else:
                # if the subtree is a leaf, return the label/leaf
                return subtree

    return 0 # Return default value when no matching node is found


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    # INSERT YOUR CODE HERE
    return (1/len(y_true)) * sum(y_true!=y_pred)
    raise Exception('Function not yet implemented!')

def compute_confusion_matrix(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    # INSERT YOUR CODE HERE
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range (len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                true_positive += 1
            else:
                false_negative += 1
        else: # y_true[i] == 0
            if y_pred[i] == 1:
                false_positive += 1
            else:
                true_negative += 1
    
    return np.array([[true_positive, false_negative],
                     [false_positive, true_negative]])

    

    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):

    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def part_b_function(Xtrn, ytrn, Xtst, ytst):
    print("----------- PART B: Train Trees for max_depths 1 and 2 -----------")
    for max_depth in range(1,3):
        print(f"Training tree with max_depth = {max_depth}")

        # Learn a decision tree
        decision_tree = id3(Xtrn, ytrn, max_depth=max_depth)
        visualize(decision_tree, max_depth)

        # Compute testing error
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred_tst)
        testing_errors.append(tst_err)
        print(f"Binary Confusion Matrix for max_depth = {max_depth}")
        print(compute_confusion_matrix(ytst, y_pred_tst))
        print()

def part_c_function(Xtrn, ytrn, Xtst, ytst, name: str):
    print("----------- PART C: Train a scikit tree -----------")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrn, ytrn)
    y_pred_tst = clf.predict(Xtst)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data) 
    graph.render(name)
    # visualize(clf, clf.get_depth())
    print(f"Visualized Decision Tree stored in: {name}.pdf")
    print(f"Binary Confusion Matrix for max_depth = {clf.get_depth()}")
    print(compute_confusion_matrix(ytst, y_pred_tst))


if __name__ == '__main__':

# Split Train-Test

    df = pd.read_csv('data_processed2.csv')
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    from sklearn.model_selection import train_test_split
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    Xtrn = Xtrn.to_numpy()
    Xtst = Xtst.to_numpy()
    ytrn = ytrn.to_numpy()
    ytst = ytst.to_numpy()




# ASSIGNMENT PART A
    print("----------- PART A: Train Monks Datasets for different max_depths -----------")

    # # Load the training data
    # M = np.genfromtxt(f'{dataset}.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytrn = M[:, 0]
    # Xtrn = M[:, 1:]

    # # Load the test data
    # M = np.genfromtxt(f'{dataset}.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytst = M[:, 0]
    # Xtst = M[:, 1:]

    # Initialize lists to store training and testing errors
    training_errors = []
    testing_errors = []
    depths = range(1, 11)  # Depths from 1 to 10

    # Loop through different tree depths
    for max_depth in depths:
        print(f"Training tree with max_depth = {max_depth}")

        # Learn a decision tree
        decision_tree = parallel_id3(Xtrn, ytrn, max_depth=max_depth)
        print()
        # visualize(decision_tree, max_depth)

        # Compute training error
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred_trn)
        training_errors.append(trn_err)

        # Compute testing error
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred_tst)
        testing_errors.append(tst_err)
        print(f"Train error: {trn_err}")
        print(f"Test error: {tst_err}")
        print(f"Binary Confusion Matrix for max_depth = {max_depth}")
        print(compute_confusion_matrix(ytst, y_pred_tst))
        print()
    print(training_errors)
    print(testing_errors)

    #     # Plotting the learning curves
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(depths, training_errors, marker='o', label='Training Error')
    #     plt.plot(depths, testing_errors, marker='o', label='Testing Error')
    #     plt.xlabel('Tree Depth')
    #     plt.ylabel('Error')
    #     plt.title(f'Learning Curves for {dataset}')
    #     plt.xticks(depths)
    #     plt.legend()
    #     plt.grid(True)
    #  #   plt.savefig(f'{dataset_name}_learning_curves.png')  # Save plot to file
    #     plt.show()