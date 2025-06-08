import random

# 爬山演算法主體
def hillClimbing(x, height, neighbor, max_fail=10000):
    fail = 0
    while True:
        nx = neighbor(x)  # 取得鄰近解
        if height(nx) > height(x):  
            x = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail:  # 若失敗次數過多則停下
                return x

# 產生鄰近解（每個維度微調）
def neighbor(x, h=0.01):
    return [xi + random.uniform(-h, h) for xi in x]

# 評估函數值（注意取負數是為了爬山法找最大，實際上我們想找最小值）
def height(x):
    # 原始函數為 f(x, y, z) = x^2 + y^2 + z^2 - 2x - 4y - 6z + 8
    # 我們用 -f(x, y, z) 來讓爬山演算法找到最小值
    return -(x[0]**2 + x[1]**2 + x[2]**2 - 2*x[0] - 4*x[1] - 6*x[2] + 8)

# 初始點
x = hillClimbing([0, 0, 0], height, neighbor)

# 輸出結果（記得再反過來還原原本的函數值）
print("x = {:.3f}, y = {:.3f}, z = {:.3f}, f(x, y, z) = {:.3f}".format(
    x[0], x[1], x[2], -height(x)
))