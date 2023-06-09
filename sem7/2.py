# Задача 2. Исследовалось влияние препарата на уровень давления пациентов. Сначала
# измерялось давление до приема препарата, потом через 10 минут и через 30 минут. Есть
# ли статистически значимые различия между измерениями давления? В выборках не соблюдается условие нормальности.
# 1е измерение до приема препарата: 150, 160, 165, 145, 155
# 2е измерение через 10 минут: 140, 155, 150, 130, 135
# 3е измерение через 30 минут: 130, 130, 120, 130, 125

# Группы зависимые и выборки множественные: критерий Фридмана.

import numpy as np 
import scipy.stats as stats
x1=[150, 160, 165, 145, 155]
x2=[140, 155, 150, 130, 135]
x3=[130, 130, 120, 130, 125]
stats.friedmanchisquare(x1,x2,x3)
print(stats.friedmanchisquare(x1,x2,x3))

# FriedmanchisquareResult(statistic=9.578947368421062, pvalue=0.00831683351100441)
# Т.к. pvalue=0.008 < a=0.05, то мы отвергаем гипотезу Н0. Это означает, что статистические различия между измерениями давления
# имеются и препарат работает.
