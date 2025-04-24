
# Ensure matplotlib is installed by running: pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess data
digits = load_digits()
X = digits.data / 16.0  # Normalize to 0–1
y = digits.target.reshape(-1, 1)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)

y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model parameters
input_size = X.shape[1]  # 64
hidden_size = 32
output_size = 10

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward and backward pass
def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backward(X, y, z1, a1, z2, a2, lr=0.1):
    global W1, b1, W2, b2
    m = y.shape[0]
    dz2 = a2 - y
    dW2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = (dz2 @ W2.T) * relu_derivative(z1)
    dW1 = X.T @ dz1 / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# Train for 300 epochs
for epoch in range(1000):
    z1, a1, z2, a2 = forward(X_train)
    backward(X_train, y_train, z1, a1, z2, a2)

# Predict on test set
_, _, _, test_pred = forward(X_test)
true_labels = np.argmax(y_test, axis=1)
pred_labels = np.argmax(test_pred, axis=1)

# Visualize predictions on 10 random samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
indices = np.random.choice(len(X_test), 10, replace=False)

for ax, idx in zip(axes.ravel(), indices):
    image = X_test[idx].reshape(8, 8)
    true_label = true_labels[idx]
    pred_label = pred_labels[idx]
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color='green' if pred_label == true_label else 'red')
    ax.axis('off')

plt.tight_layout()
plt.suptitle("Digit Predictions (Test Set)", fontsize=16, y=1.05)
plt.show()
