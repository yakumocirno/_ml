import torch
import matplotlib.pyplot as plt

# ✅ 使用 AI 工具產生資料（這裡用 torch + 一點 noise 模擬）
# 真實參數：y = 2x + 3
torch.manual_seed(42)
X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
Y = 2 * X + 3 + 0.5 * torch.randn(X.size())  # 加一些 noise 模擬實際資料

# 建立模型參數：w, b
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 超參數
lr = 0.01
epochs = 200

# 訓練迴圈
for epoch in range(epochs):
    # 預測值
    y_pred = w * X + b
    # 損失函數：MSE
    loss = torch.mean((Y - y_pred) ** 2)

    # 清除梯度
    if w.grad: w.grad.zero_()
    if b.grad: b.grad.zero_()

    # 反向傳播
    loss.backward()

    # 更新參數
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 每 20 輪印一次 loss
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d} | Loss = {loss.item():.6f} | w = {w.item():.4f}, b = {b.item():.4f}")

# 最後結果
print("\n✅ 訓練完成")
print(f"學到的模型：y = {w.item():.4f} * x + {b.item():.4f}")

plt.scatter(X.numpy(), Y.numpy(), label='data')
plt.plot(X.numpy(), y_pred.detach().numpy(), color='red', label='fit')
plt.legend()
plt.title("Linear Regression with PyTorch")
plt.show()


