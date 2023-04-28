# Задача 2. В результате 10 независимых измерений некоторой величины X, выполненных с одинаковой точностью, получены опытные данные:
# 6.9, 6.1, 6.2, 6.8, 7.5, 6.3, 6.4, 6.9, 6.7, 6.1
# Предполагая, что результаты измерений подчинены нормальному закону распределения вероятностей, оценить истинное значение величины X при помощи доверительного 
# интервала, покрывающего это значение с доверительной вероятностью 0,95.

import math
from scipy import stats

n = 10
p = 0.95
a = 1-p
a = 0.05
df = n-1
df = 9
t_a = 2.62 # квантиль, найденный по таблице для t a/2
arr = [6.9, 6.1, 6.2, 6.8, 7.5, 6.3, 6.4, 6.9, 6.7, 6.1]
X = sum(arr)/n
arr.sort()
arr2 = []
for i in arr: 
    arr2.append((i-X)**2)

std = math.sqrt(sum(arr2)/(n-1)) # среднее квадратичное отклонение (несмещенная)
print(std)

t1 = X+t_a*(std/n**0.5)
t2 = X-t_a*(std/n**0.5)
print(t1)
print(f"истинное значение величины X лежит в  доверительном интервале ({t2} , {t1})")

# Истинное значение величины X лежит в  доверительном интервале (6.963496803496659 , 6.216503196503339)




