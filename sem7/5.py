# Задача 5. Заявляется, что партия изготавливается со средним арифметическим 2,5 см. Проверить
# данную гипотезу, если известно, что размеры изделий подчинены нормальному закону
# распределения. Объем выборки 10, уровень статистической значимости 5%
# 2.51, 2.35, 2.74, 2.56, 2.40, 2.36, 2.65, 2.7, 2.67, 2.34

import math

from scipy import stats
M=2.5
n=10
a=0.05
arr=[2.51, 2.35, 2.74, 2.56, 2.40, 2.36, 2.65, 2.7, 2.67, 2.34]
X = sum(arr)/n
arr2 = []
for i in arr: 
    arr2.append((i-X)**2)

std = math.sqrt(sum(arr2)/(n-1)) # среднее квадратичное отклонение (несмещенное)
print(std)

t_n=(X-M)/(std/n**0.5)
t1 = stats.t.ppf(1-a/2,df=n-1)
t2= stats.t.ppf(a/2,df=n-1)
print(f"t наблюдаемое лежит внутри интервала {t2} {t_n} {t1}, поэтому гипотеза Н0 не отвергается.")
# t наблюдаемое лежит внутри интервала -2.262157162740992 0.5630613661802959 2.2621571627409915, поэтому гипотеза Н0 не отвергается.

