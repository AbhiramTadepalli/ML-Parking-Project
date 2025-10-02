import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def partition(x):
    unique_values, inverse_indices = np.unique(x, return_inverse=True)
    partitions = {}
    for i, value in enumerate(unique_values):
        partitions[value] = np.where(inverse_indices == i)[0]
    return partitions

def entropy(y, weights=None):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def mutual_information(x, y, weights=None):
    entropy_y = entropy(y)
    values, counts = np.unique(x, return_counts=True)
    weighted_entropy = 0
    partitions = partition(x)
    for value, count in zip(values, counts):
        p_value = count / len(x)
        indices = partitions[value]
        weighted_entropy += p_value * entropy(y[indices])
    return entropy_y - weighted_entropy

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
    # If first call, generate attribute-value pairs just once
    if attribute_value_pairs is None:
        attribute_value_pairs = [
            (i, val) 
            for i in range(x.shape[1]) 
            for val in np.unique(x[:, i])
        ]

    # If all labels are the same, return the label
    if np.all(y == y[0]):
        return y[0]

    # If no more attributes or max depth reached, return majority label
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return np.bincount(y.astype(int)).argmax()

    # Cache entropy of current label set
    current_entropy = entropy(y)

    # Choose best attribute-value pair by max information gain
    best_pair = None
    max_info_gain = -np.inf
    best_split = None

    for pair in attribute_value_pairs:
        col_idx, val = pair
        match = x[:, col_idx] == val

        if np.all(match) or np.all(~match):
            continue  # Skip useless splits

        mi = current_entropy - (
            np.mean(match) * entropy(y[match]) + 
            np.mean(~match) * entropy(y[~match])
        )

        if mi > max_info_gain:
            max_info_gain = mi
            best_pair = pair
            best_split = match

    if best_pair is None:
        return np.bincount(y.astype(int)).argmax()

    # Prepare for recursive calls
    remaining_pairs = [pair for pair in attribute_value_pairs if pair != best_pair]
    node = {}
    attr_idx, attr_val = best_pair

    # False branch
    node[(attr_idx, attr_val, False)] = id3(
        x[~best_split], y[~best_split], remaining_pairs, depth+1, max_depth
    )
    # True branch
    node[(attr_idx, attr_val, True)] = id3(
        x[best_split], y[best_split], remaining_pairs, depth+1, max_depth
    )

    return node

def bootstrap_sampler(x, y, num_samples):
    indexes = np.random.choice(len(x), size=num_samples, replace=True)
    return x[indexes], y[indexes]

def bagging(x, y, max_depth=5, num_trees=10):
    ensemble = []
    for _ in range(num_trees):
        boot_x, boot_y = bootstrap_sampler(x, y, len(x))
        tree = id3(boot_x, boot_y, max_depth=max_depth)
        ensemble.append(tree)
    return ensemble

def predict_example(x, tree):
    if not isinstance(tree, dict):
        return tree
    for criterion, subtree in tree.items():
        index, value, result = criterion
        if (x[index] == value) == result:
            return predict_example(x, subtree)
    return 0

def predict_ensemble(x, ensemble):
    predictions = [predict_example(x, tree) for tree in ensemble]
    return np.bincount(predictions).argmax()


if __name__ == '__main__':
    # Load and prepare data
    data = pd.read_csv(r"data_processed2.csv")
    X = data.drop(columns=['label']).values
    y = data['label'].values

    # Step 1: Randomly sample 5% of the data for use
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=0.05, random_state=42, stratify=y
    )

    # Step 2: Split that 5% into train and test sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
    )

    # Train bagged ensemble
    bagged_trees = bagging(X_train, y_train, max_depth=5, num_trees=10)

    # Evaluate
    y_pred = [predict_ensemble(x, bagged_trees) for x in X_test]
    accuracy = np.mean(y_pred == y_test)
    print(f"Bagging Accuracy (using 5% of data for both train and test): {accuracy:.4f}")

    from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted    0     1")
    print(f"Actual  0  {cm[0, 0]:5d} {cm[0, 1]:5d}")
    print(f"        1  {cm[1, 0]:5d} {cm[1, 1]:5d}")

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=["0", "1"], output_dict=True)
    print("\nClassification Report:")
    print(f"{'Class':>7} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label in ["0", "1"]:
        row = report[label]
        print(f"{label:>7} {row['precision']:10.2f} {row['recall']:10.2f} {row['f1-score']:10.2f} {int(row['support']):10d}")

    # Probability predictions for ROC (averaging tree outputs)
    y_prob = [
        np.mean([predict_example(x, tree) for tree in bagged_trees])
        for x in X_test
    ]

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print TPR and FPR at threshold 0.5
    threshold_index = np.argmin(np.abs(thresholds - 0.5))
    print(f"\nTPR (Sensitivity) at threshold 0.5: {tpr[threshold_index]:.4f}")
    print(f"FPR at threshold 0.5: {fpr[threshold_index]:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Compute TPR, FPR, and ROC curve
    # Assume binary classification with labels {0, 1}
    y_prob = [
        np.mean([predict_example(x, tree) for tree in bagged_trees])
        for x in X_test
    ]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print TPR and FPR at threshold 0.5
    threshold_index = np.argmin(np.abs(thresholds - 0.5))
    print(f"TPR (sensitivity) at threshold 0.5: {tpr[threshold_index]:.4f}")
    print(f"FPR at threshold 0.5: {fpr[threshold_index]:.4f}")

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
