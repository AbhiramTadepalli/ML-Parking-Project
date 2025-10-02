import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ----------- Neural Network Implementation -----------

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize neural network with specified architecture and learning rate
        
        Args:
            layer_sizes: List containing the number of neurons in each layer
            learning_rate: Step size for gradient descent (default: 0.01)
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.parameters = {}
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases using He initialization"""
        np.random.seed(42)  # For reproducibility
        
        for l in range(1, len(self.layer_sizes)):
            # He initialization for better training with ReLU
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_sizes[l-1], self.layer_sizes[l]
            ) * np.sqrt(2. / self.layer_sizes[l-1])
            
            # Initialize biases with zeros
            self.parameters[f'b{l}'] = np.zeros((1, self.layer_sizes[l]))
    
    def _relu(self, Z):
        """ReLU activation function: max(0, Z)"""
        return np.maximum(0, Z)
    
    def _sigmoid(self, Z):
        """Sigmoid activation function: 1/(1+e^(-Z))"""
        return 1 / (1 + np.exp(-Z))
    
    def _forward_prop(self, X):
        """
        Forward propagation through the network
        
        Args:
            X: Input features
            
        Returns:
            cache: Dictionary containing activations and pre-activations
        """
        cache = {'A0': X}
        L = len(self.layer_sizes) - 1
        
        # Hidden layers with ReLU activation
        for l in range(1, L):
            cache[f'Z{l}'] = np.dot(cache[f'A{l-1}'], self.parameters[f'W{l}']) + self.parameters[f'b{l}']
            cache[f'A{l}'] = self._relu(cache[f'Z{l}'])
        
        # Output layer with sigmoid activation for binary classification
        cache[f'Z{L}'] = np.dot(cache[f'A{L-1}'], self.parameters[f'W{L}']) + self.parameters[f'b{L}']
        cache[f'A{L}'] = self._sigmoid(cache[f'Z{L}'])
        
        return cache
    
    def _compute_cost(self, AL, Y, class_weights=None):
        """
        Compute weighted binary cross-entropy loss
        
        Args:
            AL: Output of the network
            Y: True labels
            class_weights: Dictionary mapping class labels to weights
            
        Returns:
            cost: Weighted binary cross-entropy loss
        """
        m = Y.shape[0]
        
        # Default equal weights if none provided
        if class_weights is None:
            weights = np.ones_like(Y)
        else:
            # Apply class weights based on true labels
            weights = np.zeros_like(Y)
            for class_label, weight in class_weights.items():
                weights[Y == class_label] = weight
        
        # Weighted binary cross-entropy with epsilon for numerical stability
        cost = (-1/m) * np.sum(
            weights * (Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
        )
        
        return np.squeeze(cost)
    
    def _backward_prop(self, cache, X, Y, class_weights=None):
        """
        Backward propagation to compute gradients
        
        Args:
            cache: Values from forward propagation
            X: Input features
            Y: True labels
            class_weights: Dictionary mapping class labels to weights
            
        Returns:
            gradients: Dictionary containing gradients for weights and biases
        """
        m = X.shape[0]
        L = len(self.layer_sizes) - 1
        gradients = {}
        
        # Apply weights to the gradient
        if class_weights is None:
            weights = np.ones_like(Y)
        else:
            weights = np.zeros_like(Y)
            for class_label, weight in class_weights.items():
                weights[Y == class_label] = weight
        
        # Weighted gradient for output layer
        dAL = weights * (cache[f'A{L}'] - Y)  # Weighted derivative
        gradients[f'dW{L}'] = (1/m) * np.dot(cache[f'A{L-1}'].T, dAL)
        gradients[f'db{L}'] = (1/m) * np.sum(dAL, axis=0, keepdims=True)
        
        dA_prev = np.dot(dAL, self.parameters[f'W{L}'].T)
        
        # Backpropagate through hidden layers
        for l in reversed(range(1, L)):
            # ReLU derivative: 1 if Z > 0, 0 otherwise
            dZ = dA_prev * (cache[f'Z{l}'] > 0).astype(float)
            gradients[f'dW{l}'] = (1/m) * np.dot(cache[f'A{l-1}'].T, dZ)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            if l > 1:
                dA_prev = np.dot(dZ, self.parameters[f'W{l}'].T)
        
        return gradients
    
    def _update_parameters(self, gradients):
        """Update parameters using gradient descent"""
        L = len(self.layer_sizes) - 1
        
        for l in range(1, L+1):
            self.parameters[f'W{l}'] -= self.learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * gradients[f'db{l}']
    
    def train(self, X, Y, epochs=1000, print_cost=True, class_weights=None):
        """
        Train the neural network
        
        Args:
            X: Training features
            Y: Training labels
            epochs: Number of training iterations
            print_cost: Whether to print cost during training
            class_weights: Dictionary mapping class labels to weights
        """
        # Use tqdm for a nice progress bar
        for i in tqdm(range(epochs)):
            cache = self._forward_prop(X)
            cost = self._compute_cost(cache[f'A{len(self.layer_sizes)-1}'], Y, class_weights)
            gradients = self._backward_prop(cache, X, Y, class_weights)
            self._update_parameters(gradients)
            
            # Print cost every 100 epochs if requested
            if print_cost and i % 100 == 0:
                print(f"Cost after epoch {i}: {cost:.4f}")
    
    def predict(self, X, threshold=0.3):
        """
        Make binary predictions
        
        Args:
            X: Input features
            threshold: Classification threshold (default: 0.3)
            
        Returns:
            Binary predictions (0 or 1)
        """
        cache = self._forward_prop(X)
        AL = cache[f'A{len(self.layer_sizes)-1}']
        return (AL > threshold).astype(int)
    
    def predict_proba(self, X):
        """Get probability predictions without threshold"""
        cache = self._forward_prop(X)
        return cache[f'A{len(self.layer_sizes)-1}']


def compute_confusion_matrix(y_true, y_pred):
    """
    Computes a confusion matrix between true labels and predicted labels.
    
    Returns a 2x2 numpy array with format:
    [[true_positive, false_negative],
     [false_positive, true_negative]]
    """
    # Flatten arrays if they're 2D
    y_true = y_true.flatten() if len(y_true.shape) > 1 else y_true
    y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:  # y_true[i] == 0
            if y_pred[i] == 1:
                false_positive += 1
            else:
                true_negative += 1
    
    return np.array([[true_positive, false_negative],
                     [false_positive, true_negative]])


# ----------- Main Execution -----------

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(r"data_processed2.csv")
    
    # Prepare features and label
    X = df.drop(columns=['label']).values.astype(float)
    y = df['label'].values.reshape(-1, 1)
    
    # Check class distribution
    print(f"Class counts:\n{pd.Series(y.flatten()).value_counts()}")
    
    # Standardize features (zero mean, unit variance)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Take a small sample for faster training (just for development)
    sample_size = int(X_train.shape[0] * 0.01)  # Using 1% of data for quick testing
    random_state = np.random.RandomState(42)
    indices = random_state.choice(X_train.shape[0], size=sample_size, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # Define network architecture
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 16, 8, 1]  # Input layer, 2 hidden layers, output layer
    
    # Initialize and train neural network with class weights to handle imbalance
    class_weights = {0: 1, 1: 5}  # Weight positive class 5x more than negative
    nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=0.01)
    nn.train(X_train, y_train, epochs=1000, class_weights=class_weights)
    
    # Make predictions
    train_preds = nn.predict(X_train)
    test_preds = nn.predict(X_test)
    
    # Basic evaluation
    print(f"\nTraining Accuracy: {np.mean(train_preds == y_train)*100:.2f}%")
    print(f"Test Accuracy: {np.mean(test_preds == y_test)*100:.2f}%")
    
    # Show confusion matrices
    print(f"Confusion Matrix (Train):\n{compute_confusion_matrix(y_train, train_preds)}")
    print(f"Confusion Matrix (Test):\n{compute_confusion_matrix(y_test, test_preds)}")
    
    # Show some example predictions
    print("\nSample predictions (test set):")
    for i in range(min(5, len(y_test))):
        print(f"True: {y_test[i][0]}, Predicted: {test_preds[i][0]}")
    
    # ----------- Detailed Evaluation -----------
    
    # Get probability predictions
    y_prob = nn.predict_proba(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)  # Using 0.5 threshold for final evaluation
    y_test_flat = y_test.flatten()
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test_flat)
    print(f"\nFinal Neural Network Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_flat, y_pred)
    print("\nConfusion Matrix:")
    print("        Predicted")
    print("        0     1")
    print(f"Actual 0 {cm[0, 0]:5d} {cm[0, 1]:5d}")
    print(f"       1 {cm[1, 0]:5d} {cm[1, 1]:5d}")
    
    # Classification Report
    report = classification_report(y_test_flat, y_pred, target_names=["0", "1"], output_dict=True)
    print("\nClassification Report:")
    print(f"{'Class':>7} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label in ["0", "1"]:
        row = report[label]
        print(f"{label:>7} {row['precision']:10.2f} {row['recall']:10.2f} {row['f1-score']:10.2f} {int(row['support']):10d}")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test_flat, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Print performance metrics at threshold 0.5
    threshold_index = np.argmin(np.abs(thresholds - 0.5))
    print(f"\nTPR (Sensitivity) at threshold 0.5: {tpr[threshold_index]:.4f}")
    print(f"FPR at threshold 0.5: {fpr[threshold_index]:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
