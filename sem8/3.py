# Задача 3 Известно, что рост футболистов в сборной распределен нормально с дисперсией генеральной совокупности, равной 25 кв.см.
# Объем выборки равен 27, среднее выборочное составляет 174.2. Найдите доверительный интервал для математического ожидания с надежностью 0.95.

d = 25
p = 0.95
a = 0.05
n = 27
M = 174.2
Za = 1.96 # квантиль, найденный по таблице для z alpha/2

z1 = M+Za*(d**0.5/n**0.5)
z2 = M-Za*(d**0.5/n**0.5)

print(f" Доверительный интервал для математического ожидания с надежностью 0.95: [{z2}, {z1}]")

#  Доверительный интервал для математического ожидания с надежностью 0.95: [172.31398912064722, 176.08601087935276]