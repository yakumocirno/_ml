##  專案目標

透過簡單的物理判斷策略（手寫公式），讓竿子在車上盡可能保持平衡而不倒下。

---

##  使用技術

- Python
- [Gymnasium]：強化學習環境模擬套件
- 無需 AI、無需模型訓練

---

##  程式邏輯說明

```python
cart_pos, cart_vel, pole_angle, pole_vel = observation
action = 1 if pole_angle + 0.5 * pole_vel > 0 else 0