# Задача 2. Проведите тест гипотезы. Утверждается, что шарики для подшипников, изготовленные автоматическим станком, имеют средний диаметр 17 мм.
# Используя односторонний критерий с α=0,05, проверить эту гипотезу, если в выборке из n=100 шариков средний диаметр оказался равным 17.5 мм, 
# а дисперсия известна и равна 4 кв. мм.

# 1. H_0: M = 17 мм
# H1: M > 17 # ПКО

# 2. alpha = 0.05
# 1-alpha = 0.95
# Z_1-alpha = 1,64 #  по таблице




D = 4
std=D**0.5
X = 17.5
M = 17
n = 100
Z_n=(X-M)/(std/n**0.5)

print(Z_n)

D=4
std=D**0.5
n=10
X=17.5
a=0.05
m=17
Zt=1.64# от табличного значения при 1-a
Zn=(X-m)/(std/n**0.5)
print(f"значение наблюдаемого Z = {Zn}")