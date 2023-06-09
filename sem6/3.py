# Задача 3. Рост дочерей 175, 167, 154, 174, 178, 148, 160, 167, 169, 170
# Рост матерей 178, 165, 165, 173, 168, 155, 160, 164, 178, 175
# Используя эти данные построить 95% доверительный интервал для разности среднего роста родителей и детей.


n = 10
p = 0.95
a = 1-p # a = 0.05
df = 2*(n-1) # df = 18
t_a = 2.101 # квантиль, найденный по таблице для t a/2
arr1 = [175, 167, 154, 174, 178, 148, 160, 167, 169, 170]
arr2 = [178, 165, 165, 173, 168, 155, 160, 164, 178, 175]
X1 = sum(arr1)/n
X2 = sum(arr2)/n
arr3 = []
arr4 = []
for i in arr1: 
    arr3.append((i-X1)**2)
D1 = sum(arr3)/(n-1) 
for i in arr2: 
    arr4.append((i-X2)**2)
D2 = sum(arr4)/(n-1) 

t1 = (X2-X1)+t_a*((D1/n+D2/n)**0.5)
t2 = (X2-X1)-t_a*((D1/n+D2/n)**0.5)
print(f"доверительный интервал для разности среднего роста родителей и детей ({t2} , {t1})")

# доверительный интервал для разности среднего роста родителей и детей (-6.268721143279807 , 10.068721143279818)