旅行推銷員問題（TSP）：  
給定一組城市座標，要求出一條**最短的巡迴路徑**，使推銷員能走訪每座城市一次，最後回到起點。

---

- 初始隨機路徑
- 每次隨機交換兩個城市（產生鄰近解）
- 若新路徑更短則接受，否則累積失敗次數
- 連續多次無改善則停止（局部最佳）

---

| 函式名稱         | 功能說明                            |
|------------------|-------------------------------------|
| `distance`       | 計算兩城市間的歐氏距離              |
| `total_distance` | 計算整條巡迴路徑的總長度            |
| `neighbor`       | 隨機交換路徑中兩個城市              |
| `hillClimbing`   | 核心爬山演算法流程                  |

最佳路徑順序: [3, 2, 0, 6, 7, 4, 5, 1, 9, 8]
最短距離: 410.7821