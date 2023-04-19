# Задача 2. В первом ящике находится 8 мячей, из которых 5 - белые. Во втором ящике - 12 мячей, из которых 5 белых. Из первого
# ящика вытаскивают случайным образом два мяча, из второго - 4 мяча. Какова вероятность того, что 3 мяча белые?

import math
def Comb(n,k):
    return int(math.factorial(n)/math.factorial(k)*math.factorial(n-k))

p0 = Comb(5,2)/Comb(8,2) # вероятность достать 2 белых мяча из 1 ящика
p1 = Comb(3,2)/Comb(8,2) # вероятность достать 2 черных мяча из 1 ящика
p2 = Comb(5,1)*Comb(3,1)/Comb(8,2) # вероятность достать 1 белый и 1 черный мяч из 1 ящика
p3 = Comb(5,3)*Comb(7,1)/Comb(12,4) # вероятность достать 3 белых и 1 черный мяч из 2 ящика
p4 = Comb(5,2)*Comb(7,2)/Comb(12,4) # вероятность достать 2 белых и 2 черных мяча из 2 ящика
p5 = Comb(5,1)*Comb(7,3)/Comb(12,4) # вероятность достать 1 белый и 3 черных мяча из 2 ящика

p_0 = p0*p5 #  вероятность достать 2 белых мяча из 1 ящика и 1 белый из 2 ящика
p_1 = p2*p5 # вероятность достать 1 белый мяч из 1 ящика и 2 белых из 2-го
p_2 = p1*p3 # вероятность достать 3 белых мяча из 1 ящика и 2 черных из 2-го

p = p_0+p_1+p_2 # общая вероятность
