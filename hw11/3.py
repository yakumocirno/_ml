from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + 5 + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("✅ 回歸結果：")
print(f"R2 = {r2_score(y_test, y_pred):.4f}, MSE = {mean_squared_error(y_test, y_pred):.4f}")

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, color='red', label="Prediction")
plt.title("回歸線擬合")
plt.legend()
plt.show()