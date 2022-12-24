import math
import numpy as np

def read_file_data(filepath):
	file = open(filepath, "r")
	lines = file.readlines()
	n = len(lines)
	data = np.zeros(n)
	for i in range(n):
		data[i] = float(lines[i].replace(',','.'))
	file.close()
	return data, n


def F_emp(x, data, n):
	for i in range(n):
		if x < data[i]:
			return i / n
	return 1

# D+
def Smirnov_stat1(data1, data2, n):
	max = 0
	for i in range(n):
		tmp = (i + 1 ) / n - F_emp(data1[i], data2, n)
		if tmp > max:
			max = tmp
	return max
	
# D-
def Smirnov_stat2(data1, data2, n):
	max = 0
	for i in range(n):
		tmp = F_emp(data1[i], data2, n) - i / n
		if tmp > max:
			max = tmp
	return max

# Итоговая статистика Смирнова
def Smirnov_crit(data1, data2, n, k):
	stat1 = Smirnov_stat1(data1, data2, n)
	stat2 = Smirnov_stat2(data1, data2, n)
	res = max(stat1, stat2) * math.sqrt(n * n / (2 * n))
	print("Статистика Смирнова:", res)
	print("Квантиль распределения Колмогорова:", k)
	if (res < k):
		print("Принимаем гипотезу")
	else:
		print("Отвергаем гипотезу")


# предполагаем что все значения различны и размеры равны
def W_stat(x1, x2, n):
	y = np.concatenate([x1, x2])
	y.sort()
	res = 0
	i = 0
	for j in range(2 * n):
		if x1[i] == y[j]:
			res += j + 1
			i += 1
	return res

def W_crit(x, y, n, w):
	stat = W_stat(x, y, n)
	seg1 = (n * (n + 1) / 2, w)
	seg2 = (n * (2 * n + 1) - w, n * n + n * (n + 1) / 2)
	print("Статистика Вилкоксона:", stat)
	print("Крит область 1:", seg1)
	print("Крит область 2:", seg2)



# читаем данные
exp, n = read_file_data("data_4_1_exp")
imp, n = read_file_data("data_4_1_imp")

# вычисляем разности и сортируем
n = n - 1
y1 = np.zeros(n)
y2 = np.zeros(n)

for i in range (n):
	y1[i] = exp[i + 1] - exp[i]
	y2[i] = imp[i + 1] - imp[i]

y1.sort()
y2.sort()

# квантиль для p = 0.05
k = 0.52
Smirnov_crit(y1, y2, n, k)

# квантиль для p / 2 = 0.025, m = n = 11
w = 96
W_crit(y1, y2, n, w)


