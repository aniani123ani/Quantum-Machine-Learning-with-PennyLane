import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.1)
X = StandardScaler().fit_transform(X)
y = np.where(y == 0, -1, 1)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(params, x):
    """Parameterized quantum circuit."""
    qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

num_layers = 3
params = np.random.uniform(-np.pi, np.pi, (num_layers, n_qubits, 3))

@qml.qnode(dev)
def model(params, x):
    return quantum_circuit(params, x)

def predict(params, X):
    return np.sign([model(params, x) for x in X])

# Step 4: Cost function and optimizer
def cost(params, X, y):
    predictions = [model(params, x) for x in X]
    return np.mean((np.array(predictions) - y) ** 2)

optimizer = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 50
costs = []

for step in range(steps):
    params = optimizer.step(lambda p: cost(p, X_train, y_train), params)
    current_cost = cost(params, X_train, y_train)
    costs.append(current_cost)
    if step % 10 == 0:
        print(f"Step {step} - Cost: {current_cost}")

y_pred = predict(params, X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Training Convergence")
plt.grid()
plt.show()

def plot_decision_boundary(params, X, y):
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict(params, grid).reshape(xx.shape)
    plt.contourf(xx, yy, preds, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(params, X, y)
