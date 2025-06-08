import torch

# 建立變數（requires_grad=True 才能追蹤梯度）
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

# 超參數
lr = 0.1
max_iter = 100

for step in range(max_iter):
    # 清除之前的梯度
    if x.grad: x.grad.zero_()
    if y.grad: y.grad.zero_()
    if z.grad: z.grad.zero_()

    # 定義函數 f(x, y, z)
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 自動反向傳播
    f.backward()

    # 使用梯度下降法更新變數
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
        z -= lr * z.grad

    # 顯示每10步的狀態
    if step % 10 == 0 or step == max_iter - 1:
        print(f"Step {step:03d} | f = {f.item():.6f} | x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")

print("\n✅ 最小值結果：")
print(f"x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}, f(x, y, z) = {f.item():.6f}")