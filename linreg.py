import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class SimpleLinearRegression():
    # A simple Linear Regression Model using Gradient Descent
    def __init__(self):
        self.w = []  # Model weights (includes bias term)
    
    def initialize_weights(self, num_features):
        # Initialize weights to zeros
        return np.zeros(num_features)
    
    def compute_loss(self, X, y):
        # Compute Mean Squared Error (MSE) loss
        X = self.prepend_ones(X)
        y_pred = np.dot(X, self.w)
        return (1 / (2 * len(y))) * np.sum((y_pred - y) ** 2)

    def predict_example(self, w, x):
        # Predict output for a single example
        return np.dot(w, x)

    def gradient_descent(self, w, X, y, lr):
        # Perform one step of gradient descent
        n = len(y)
        y_pred = np.dot(X, w)
        error = y_pred - y
        gradient = (1 / n) * np.dot(X.T, error)  # Correct gradient
        gradient = np.clip(gradient, -1e2, 1e2)  # Optional: Clip for stability
        return w - lr * gradient
        
    def prepend_ones(self, X):
        # Add bias term column (for intercept)
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X, y, lr=0.1, iters=100, recompute=True):
        # Fit the model using gradient descent
        X = self.prepend_ones(X)
        y = np.array(y) 

        # Re-initialize weights if recompute is True
        if recompute:
            self.w = self.initialize_weights(X.shape[1])

        # Gradient Descent loop
        for _ in range(iters):
            self.w = self.gradient_descent(self.w, X, y, lr)

    def predict_labels(self, X):
        # Predict outputs for all examples
        X = self.prepend_ones(X)
        return np.array([self.predict_example(self.w, x) for x in X])

    @staticmethod
    def compute_error(y_true, y_pred):
        # Compute Mean Squared Error for predictions
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

if __name__ == '__main__':
    # Load dataset
    df = pd.read_csv(r"data_processed3.csv")
    
    # Handle missing values
    df = df.dropna(subset=['spots_open'])  # Remove rows with missing target 'spots_open'
    df = df.fillna(df.mean(numeric_only=True))  # Fill numeric NaNs with column means
    
    # Define feature matrix (X) and target vector (y)
    X = df.drop(columns=['spots_open'])
    y = df['spots_open']

    # Split data into training and testing sets
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features and target variable using StandardScaler
    scaler_x = StandardScaler()
    Xtrn = scaler_x.fit_transform(Xtrn)
    Xtst = scaler_x.transform(Xtst)
    
    scaler_y = StandardScaler()
    ytrn = scaler_y.fit_transform(ytrn.values.reshape(-1, 1)).flatten()
    ytst = scaler_y.transform(ytst.values.reshape(-1, 1)).flatten()

    # Initialize the SimpleLinearRegression model
    lr = SimpleLinearRegression()
    
    # Training loop with inverse scaling for interpretable errors
    for iter in [100000]:
        for a in [.5]:
            lr.fit(X=Xtrn, y=ytrn, lr=a, iters=iter, recompute=True)
            
            # Make predictions and inverse transform them
            y_train_pred = scaler_y.inverse_transform(
                lr.predict_labels(Xtrn).reshape(-1, 1)
            ).flatten()
            y_test_pred = scaler_y.inverse_transform(
                lr.predict_labels(Xtst).reshape(-1, 1)
            ).flatten()
            
            # Calculate and print the errors in original scale
            train_error = mean_squared_error(
                scaler_y.inverse_transform(ytrn.reshape(-1, 1)), 
                y_train_pred
            )
            test_error = mean_squared_error(
                scaler_y.inverse_transform(ytst.reshape(-1, 1)),
                y_test_pred
            )
            
            print(f'Iterations={iter}, LR={a:.6f}, Train Error={train_error:.2f}, Test Error={test_error:.2f}')
            
            # Uncomment below to visualize training diagnostics
            '''
            plt.figure(figsize=(10,6))
            plt.scatter(scaler_y.inverse_transform(ytrn.reshape(-1,1)), 
                        y_train_pred, alpha=0.3)
            plt.plot([0,200], [0,200], 'r--')  # Ideal line
            plt.xlabel('True Spots Open')
            plt.ylabel('Predicted Spots Open')
            plt.title('Prediction Diagnostic')
            plt.show()
            '''

    # Final predictions on the test set
    predictions_scaled = lr.predict_labels(Xtst)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    true_values = scaler_y.inverse_transform(ytst.reshape(-1, 1)).flatten()

    # Print a few example results
    num_examples_to_print = 5
    print("Example | True spots_open | Predicted spots_open")
    for i in range(num_examples_to_print):
        print(f"{i+1:7} | {true_values[i]:15.2f} | {predictions[i]:19.2f}")
