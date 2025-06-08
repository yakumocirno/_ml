import sympy as sp

# 定義符號變數
x, y = sp.symbols('x y')

# --------- 第一題 ---------
# f(x, y) = x^2 * y
f1 = x**2 * y

# 計算梯度（偏微分）
grad_f1_x = sp.diff(f1, x)
grad_f1_y = sp.diff(f1, y)
print("題一：f(x, y) = x^2 * y")
print("梯度 ∇f =", [grad_f1_x, grad_f1_y])  # [2xy, x^2]

# --------- 第二題 ---------
# f(x, y) = sin(x) + y^2
f2 = sp.sin(x) + y**2

# 計算梯度
grad_f2_x = sp.diff(f2, x)
grad_f2_y = sp.diff(f2, y)
print("\n題二：f(x, y) = sin(x) + y^2")
print("梯度 ∇f =", [grad_f2_x, grad_f2_y])  