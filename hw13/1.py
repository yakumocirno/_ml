from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Gradient Boosting 回歸結果：")
print(f"MSE = {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2  = {r2_score(y_test, y_pred):.4f}")