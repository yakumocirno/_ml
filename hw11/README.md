#  機器學習三大任務範例：分類 / 分群 / 回歸（使用 scikit-learn）

本專案展示使用 Python 與 scikit-learn 套件，實作三種常見的機器學習任務：

1.  分類（Classification）
2.  分群（Clustering）
3.  回歸（Regression）

---

##  套件需求
pip install scikit-learn matplotlib numpy

1️⃣ 分類：KNN 於 IRIS 資料集
使用 IRIS 資料集（三類花種）

模型：KNN（k=3）

輸出分類報告（精確度、召回率等）

2️⃣ 分群：KMeans + make_blobs
使用 make_blobs 產生 3 群假資料

模型：KMeans（k=3）

輸出散佈圖顯示分群結果

3️⃣ 回歸：Linear Regression
手動產生線性資料 y = 2.5x + 5 + noise

模型：線性回歸

輸出 R²、MSE 評估指標

繪製預測 vs 真實資料對照圖
