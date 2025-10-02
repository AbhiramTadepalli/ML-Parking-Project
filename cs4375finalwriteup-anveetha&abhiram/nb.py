import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# ====================== PREPROCESSING ======================
def preprocess_data(df):
    """Clean and transform raw data for Naive Bayes"""
    # Remove rows with missing values
    df_clean = df.dropna()

    # Filter out 10pm–6am (time in minutes since midnight)
    night_mask = (df_clean['time'] >= 1320) | (df_clean['time'] < 360)
    df_clean = df_clean[~night_mask]

    # Define time bins and labels
    time_bins = [0, 360, 540, 720, 900, 1080, 1260, 1320, 1440]
    labels = [
        'midnight_early_morning',
        'early_morning',
        'late_morning',
        'early_afternoon',
        'late_afternoon',
        'evening',
        'late_evening',
        'night'
    ]
    df_clean['time_period'] = pd.cut(df_clean['time'], bins=time_bins, labels=labels, right=False)

    # Traffic duration bins
    traffic_bins = [0, 180, 300, 420, 540, 660, 900, np.inf]
    traffic_labels = [
        'very_fast', 'fast', 'moderate', 'busy', 'slow', 'very_slow', 'extreme'
    ]
    df_clean['traffic_level'] = pd.cut(
        df_clean['traffic_duration_sec'],
        bins=traffic_bins,
        labels=traffic_labels,
        right=False
    )

    # Select features
    features = [
        'month', 'day', 'time', 'day_of_week', 'color',
        'drive_duration_sec', 'traffic_duration_sec', 'temp', 'humidity',
        'clouds', 'visibility', 'wind_speed', 'conditions',
        'garage_ps1', 'garage_ps3', 'garage_ps4'
    ]
    
    # Convert features to string for categorical handling
    for col in features:
        df_clean[col] = df_clean[col].astype(str)

    return df_clean[features], df_clean['label']

# ====================== NAIVE BAYES ======================
class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.classes = y.unique()
        n_samples = len(y)
        self.class_probs = {cls: (y == cls).sum() / n_samples for cls in self.classes}
        self.feature_probs = {cls: {} for cls in self.classes}

        for cls in self.classes:
            cls_data = X[y == cls]
            total_cls = len(cls_data)

            for feature in X.columns:
                feature_counts = cls_data[feature].value_counts()
                n_categories = len(X[feature].unique())
                self.feature_probs[cls][feature] = {
                    val: (count + self.alpha) / (total_cls + n_categories * self.alpha)
                    for val, count in feature_counts.items()
                }

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            max_prob = -np.inf
            best_class = None

            for cls in self.classes:
                prob = np.log(self.class_probs[cls])
                for feature, value in row.items():
                    feature_prob = self.feature_probs[cls][feature].get(
                        value,
                        self.alpha / (len(X) + len(X[feature].unique()) * self.alpha)
                    )
                    prob += np.log(feature_prob)
                if prob > max_prob:
                    max_prob = prob
                    best_class = cls

            predictions.append(best_class)
        return np.array(predictions)
    
    def predict_proba(self, X):
        prob_list = []
        for _, row in X.iterrows():
            class_scores = {}
            for cls in self.classes:
                prob = np.log(self.class_probs[cls])
                for feature, value in row.items():
                    feature_prob = self.feature_probs[cls][feature].get(
                        value,
                        self.alpha / (len(X) + len(X[feature].unique()) * self.alpha)
                    )
                    prob += np.log(feature_prob)
                class_scores[cls] = prob

            # Convert log-probs to normal probabilities using softmax-like normalization
            max_log_prob = max(class_scores.values())
            exp_scores = {cls: np.exp(score - max_log_prob) for cls, score in class_scores.items()}
            total = sum(exp_scores.values())
            normalized_probs = {cls: score / total for cls, score in exp_scores.items()}
            prob_list.append(normalized_probs)

        return prob_list


# ====================== MAIN ======================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    # === Load and preprocess data ===
    df = pd.read_csv(r"nb_processed2.csv")
    X, y = preprocess_data(df)

    # === Encode categorical features numerically for SMOTE ===
    X_encoded = X.apply(LabelEncoder().fit_transform)

    # === Sample and split data ===
    X_encoded = X_encoded.sample(frac=0.5, random_state=22)
    y = y.loc[X_encoded.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # === Show original class distribution ===
    print("Original class distribution:\n", y_train.value_counts())

    # === Apply SMOTE to training set ===
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Resampled class distribution:\n", y_train_resampled.value_counts())

    # === Train custom Naive Bayes classifier ===
    nb = NaiveBayesClassifier(alpha=1)
    nb.fit(X_train_resampled, y_train_resampled)

    # === Predict and evaluate ===
    predictions = nb.predict(X_test)

    accuracy = np.mean(predictions == y_test)
    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    print(pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted']))

    def classification_report_manual(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        print("\nClassification Report:")
        print(f"{'Class':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            support = np.sum(y_true == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"{cls:>8} {precision:10.2f} {recall:10.2f} {f1:10.2f} {support:10}")

    # Calling the classification report after the confusion matrix
    classification_report_manual(y_test, predictions)

    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # Convert probabilities to array of probabilities for positive class (e.g., 'yes')
    probabilities = nb.predict_proba(X_test)
    positive_class = nb.classes[1]  # Assume binary: [negative, positive]
    y_score = np.array([p[positive_class] for p in probabilities])

    # Convert y_test to binary
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test).ravel()

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
