# Задача 1. Известно, что генеральная совокупность распределена нормально со средним квадратическим отклонением, равным 16.
# Найти доверительный интервал для оценки математического ожидания с надежностью 0.95, если выборочная средняя M = 80, а объем выборки n = 256.

M = 80
n = 256
std = 16
p = 0.95
a = 1-p
a = 0.05
Za = 1.96 # квантиль, найденный по таблице для z alpha/2

z1 = M+Za*(std/n**0.5)
z2 = M-Za*(std/n**0.5)

print(z1)
print(z2)

# доверительный интервал (78.04; 81.96). 
# p(78.04 < M < 81.96) = 0.95

