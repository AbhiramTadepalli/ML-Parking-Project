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
    
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    # INSERT YOUR CODE HERE. NTE: THIS IS A RECURSIVE FUNCTION.
    # initialize attribute_value_pairs the first time
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(x.shape[1]): # iterate through each feature (column)
            for value in np.unique(x[:, i]):
                attribute_value_pairs.append((i, value)) # for each feature, make a pair with all of its unique values

    # Check termination conditions
    if len(np.unique(y)) == 1: # entire set is pure
        return y[0]
    if len(attribute_value_pairs) == 0 or depth == max_depth: # max_depth reached or no attribute-value pairs left
        return np.bincount(y.astype(int)).argmax() # return the most common y value

    # Find best attribute-value pair
    best_pair = None
    max_mutual_info = float('-inf')

    for pair in tqdm(attribute_value_pairs, desc=f"Training for depth {depth} and max_depth {max_depth}", unit="pair"):
        # Create a boolean array where True indicates the attribute equals the value
        attribute_matches = x[:, pair[0]] == pair[1]
        
        # Calculate mutual information for this pair
        mi = mutual_information(attribute_matches, y)
        
        # Update best_pair if this pair has higher mutual information
        if mi > max_mutual_info:
            max_mutual_info = mi
            best_pair = pair
        

    if max_mutual_info < 0.01:  # Minimum information gain threshold
        return np.bincount(y.astype(int)).argmax()
    
    # Create node and partition data
    node = {}
    attribute_index, attribute_value = best_pair
    x_subset = x[:, attribute_index] == attribute_value
    
    # Recursively build left and right subtrees
    left_pairs = [pair for pair in attribute_value_pairs if pair != best_pair] # pass this in as the new attribute_value_pairs
    node[(attribute_index, attribute_value, False)] = id3(x[~x_subset], y[~x_subset], left_pairs, depth + 1, max_depth) # right subtree
    node[(attribute_index, attribute_value, True)] = id3(x[x_subset], y[x_subset], left_pairs, depth + 1, max_depth) # left subtree
    
    return node
    raise Exception('Function not yet implemented!')


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

def evaluate_decision_tree(Xtst, ytst, decision_tree):
    # Make predictions
    y_pred = np.array([predict_example(x, decision_tree) for x in Xtst])
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == ytst)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix using your existing function
    cm = compute_confusion_matrix(ytst, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted    0     1")
    print(f"Actual  0  {cm[1, 1]:5d} {cm[1, 0]:5d}")
    print(f"        1  {cm[0, 1]:5d} {cm[0, 0]:5d}")
    
    # Classification Report
    from sklearn.metrics import classification_report
    report = classification_report(ytst, y_pred, target_names=["0", "1"], output_dict=True)
    print("\nClassification Report:")
    print(f"{'Class':>7} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label in ["0", "1"]:
        row = report[label]
        print(f"{label:>7} {row['precision']:10.2f} {row['recall']:10.2f} {row['f1-score']:10.2f} {int(row['support']):10d}")
    
    # For ROC curve, we need probability estimates
    # Since your decision tree returns class labels (0 or 1), we'll use them directly
    # This is not ideal but will work for basic evaluation
    y_prob = y_pred  # Using predictions as probability estimates
    
    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    fpr, tpr, thresholds = roc_curve(ytst, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Print TPR and FPR at threshold 0.5 (or closest available)
    if 0.5 in thresholds:
        threshold_index = np.where(thresholds == 0.5)[0][0]
    else:
        threshold_index = np.argmin(np.abs(thresholds - 0.5))
    
    print(f"\nTPR (Sensitivity) at threshold 0.5: {tpr[threshold_index]:.4f}")
    print(f"FPR at threshold 0.5: {fpr[threshold_index]:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    
    # Display confusion matrix as a plot
    from sklearn.metrics import ConfusionMatrixDisplay
    
    # Convert your confusion matrix format to sklearn format
    cm_display = np.array([[cm[0, 0], cm[0, 1]], 
                           [cm[1, 0], cm[1, 1]]])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=["1", "0"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':

# Split Train-Test

    df = pd.read_csv('data_processed2.csv')
    # make X and y
    X = df.drop(columns=['label'])
    y = df['label']
    
    #split
    from sklearn.model_selection import train_test_split
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #sample
    Xtrn = Xtrn.sample(frac=0.05, random_state=42) # use only 20% of the data
    ytrn = ytrn.loc[Xtrn.index]
    # smote class balancing
    print((f"Original class distribution: {ytrn.value_counts()}"))
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    Xtrn, ytrn = smote.fit_resample(Xtrn, ytrn)
    print((f"Resampled class distribution: {ytrn.value_counts()}"))
    
    
    print(len(Xtrn))
    print(len(Xtst))
    print(len(ytrn))
    print(len(ytst))
    Xtrn = Xtrn.to_numpy()
    Xtst = Xtst.to_numpy()
    ytrn = ytrn.to_numpy()
    ytst = ytst.to_numpy()




# ASSIGNMENT PART A

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
    depths = range(8, 9)  # Depths from 1 to 10

    # Loop through different tree depths
    for max_depth in depths:
        print(f"Training tree with max_depth = {max_depth}")

        # Learn a decision tree
        decision_tree = id3(Xtrn, ytrn, max_depth=max_depth)
        print()
        visualize(decision_tree, max_depth)

        # Compute training error
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred_trn)
        training_errors.append(trn_err)

        # Compute testing error
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred_tst)
        testing_errors.append(tst_err)
        print(f"Train error: {trn_err}")
        print(f"Train Binary Confusion Matrix for max_depth = {max_depth}")
        print(compute_confusion_matrix(ytrn, y_pred_trn))
        print(f"Test error: {tst_err}")
        print(f"Test Binary Confusion Matrix for max_depth = {max_depth}")
        print(compute_confusion_matrix(ytst, y_pred_tst))
        print()

        print("LETS DO THIS ONE LAST TIOME")
        evaluate_decision_tree(Xtst, ytst, decision_tree)

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
