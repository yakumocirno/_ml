import numpy as np

# ==== 1. 資料集：輸入4-bit，輸出七段顯示器7個燈 ====
X = np.array([
    [0,0,0,0],  # 0
    [0,0,0,1],  # 1
    [0,0,1,0],  # 2
    [0,0,1,1],  # 3
    [0,1,0,0],  # 4
    [0,1,0,1],  # 5
    [0,1,1,0],  # 6
    [0,1,1,1],  # 7
    [1,0,0,0],  # 8
    [1,0,0,1]   # 9
])

Y = np.array([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,0,0,1,1]   # 9
])

# ==== 2. 初始化權重與偏差 ====
np.random.seed(0)
W = np.random.randn(4, 7) * 0.01  # 權重：輸入4 → 輸出7
b = np.zeros((1, 7))              # 偏差

# ==== 3. 函數定義 ====
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_fn(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def forward(W, b):
    return sigmoid(X @ W + b)

def loss(W, b):
    y_pred = forward(W, b)
    return loss_fn(Y, y_pred)

def numerical_gradient(f, W, b, eps=1e-4):
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            tmp = W[i,j]
            W[i,j] = tmp + eps
            l1 = f(W, b)
            W[i,j] = tmp - eps
            l2 = f(W, b)
            grad_W[i,j] = (l1 - l2) / (2 * eps)
            W[i,j] = tmp
    for j in range(b.shape[1]):
        tmp = b[0,j]
        b[0,j] = tmp + eps
        l1 = f(W, b)
        b[0,j] = tmp - eps
        l2 = f(W, b)
        grad_b[0,j] = (l1 - l2) / (2 * eps)
        b[0,j] = tmp
    return grad_W, grad_b

# ==== 4. 訓練迴圈 ====
learning_rate = 0.5
for epoch in range(3000):
    grad_W, grad_b = numerical_gradient(loss, W, b)
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss(W, b):.6f}")

output = forward(W, b)
print("\n=== 預測結果（四捨五入） ===")
print(np.round(output))
print("\n=== 原始正確答案 ===")
print(Y)