# Задача 5. Устройство состоит из трех деталей. Для первой детали вероятность выйти из строя в первый месяц равна 0.1, для 
# второй - 0.2, для третьей - 0.25. Какова вероятность того, что в первый месяц выйдут из строя: а) все детали б) только две 
# детали в) хотя бы одна деталь г) от одной до двух деталей?

p1 = 0.1
p2 = 0.2
p3 = 0.25
p_a = p1*p2*p3 # вероятность того, что в первый месяц из строя выйдут все детали
p_b = p1*p2*(1-p3)+p1*(1-p2)*p3+(1-p1)*p2*p3 # вероятность того, что в первый месяц из строя выйдут 2 детали
p_c = p1*(1-p2)*(1-p3)+(1-p1)*p2*(1-p3)+(1-p1)*(1-p2)*p3+p_b+p_a # вероятность того, что в первый месяц из строя выйдет хотябы 1 деталь
p_d = p1*(1-p2)*(1-p3)+(1-p1)*p2*(1-p3)+(1-p1)*(1-p2)*p3+p_b # вероятность того, что в первый месяц из строя выйдет от 1 до 2 деталей
print(p_a)
print(p_b)
print(p_c)
print(p_d)