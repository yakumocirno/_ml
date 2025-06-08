#  什麼是 Gradient Boosting Classifier？

Gradient Boosting Classifier 是一種強大的集成學習（Ensemble Learning）方法，常用於處理分類問題。

它的核心思想是：將多個「弱分類器（如淺層決策樹）」組合起來，每個模型依序訓練、修正上一個模型的錯誤，最終形成一個準確率很高的「強分類器」。

---

##  工作原理（流程）

1. **初始模型**  
   - 先訓練一個簡單模型，通常是一棵小決策樹。

2. **計算損失**  
   - 使用 log loss 或交叉熵損失，計算模型預測與實際標籤的誤差。

3. **導出梯度**  
   - 計算損失函數的負梯度，作為新模型的學習目標。

4. **擬合新模型**  
   - 用新的樹來學習這些誤差（負梯度）。

5. **更新預測**  
   - 將新模型的預測加到整體模型中，逐步提高準確率。

6. **重複疊加**  
   - 重複上述過程直到達到預設的樹數量或滿意的準確率。

---

##  優點

- 高準確率，Kaggle 競賽常用
- 能處理非線性與高維資料
- 可提供特徵重要性分析

---

##  缺點

- 訓練時間較長（比隨機森林慢）
- 對異常值與雜訊較敏感
- 難以並行處理（每棵樹依賴前一棵）

---

##  Python 範例（scikit-learn）
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 載入資料集
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("✅ 分類報告：")
print(classification_report(y_test, y_pred))