import random
import math

# 計算距離
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# 計算總路徑長度
def total_distance(path, cities):
    dist = 0
    for i in range(len(path)):
        city_a = cities[path[i]]
        city_b = cities[path[(i + 1) % len(path)]]  # 回到起點
        dist += distance(city_a, city_b)
    return dist

# 鄰居產生：隨機交換兩個城市
def neighbor(path):
    new_path = path[:]
    i, j = random.sample(range(len(path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# 爬山演算法
def hillClimbing(path, cities, max_fail=10000):
    fail = 0
    while True:
        current_dist = total_distance(path, cities)
        new_path = neighbor(path)
        new_dist = total_distance(new_path, cities)
        if new_dist < current_dist:
            path = new_path
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return path

# 隨機產生城市座標（10 個城市）
random.seed(42)
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]

# 初始隨機路徑
init_path = list(range(len(cities)))
random.shuffle(init_path)

# 執行爬山法
best_path = hillClimbing(init_path, cities)


print("最佳路徑順序:", best_path)
print("最短距離:", total_distance(best_path, cities))
