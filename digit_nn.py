import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load data
digits = load_digits()
X = digits.data / 16.0  # Normalize pixels (0–1)
y = digits.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model params
input_size = X_train.shape[1]      # 64 (8x8)
hidden_size = 32
output_size = 10                   # Digits 0–9

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
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # prevent overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

# Forward + Backward Pass
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

# Train loop
def train(epochs=1000):
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(X_train)
        loss = compute_loss(y_train, a2)
        acc = accuracy(y_train, a2)
        backward(X_train, y_train, z1, a1, z2, a2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
    _, _, _, test_pred = forward(X_test)
    print("Test accuracy:", accuracy(y_test, test_pred))
