import numpy as np
from math import sqrt, fabs
from scipy.stats import norm, f, t

def read_file_data(filepath):
	file = open(filepath, "r")
	lines = file.readlines()
	n = len(lines)
	data = np.zeros(n)
	for i in range(n):
		data[i] = float(lines[i].replace(',','.'))
	file.close()
	return data, n

# критерий Фишера-Снедекора
def F_test(var1, var2, n, alpha):
	F1 = f.ppf(q=alpha/2, dfn=n-1, dfd=n-1)
	F2 = f.ppf(q=1-alpha/2, dfn=n-1, dfd=n-1)
	print("квантиль alpha / 2:", F1)
	print("квантиль 1 - alpha / 2:", F2)
	stat = var1 / var2
	print("F статистика:", stat)
	if stat > F1 and stat < F2:
		print("принимаем гипотезу о равенстве дисперсий")
	else:
		print("отвергаем гипотезу о равенстве дисперсий")
	return 0


# критерий Стьюдента для проверки равенства матожиданий
# при неизвестных но равных дисперсиях
def Student_test(u1, u2, var1, var2, n, alpha):
	s = sqrt((n - 1) * (var1 + var2) / (2 * n - 2))
	phi2 = (u1 - u2) / (s * sqrt(2 / n))
	t_val = t.ppf(q=1-alpha/2, df=2*n-2)
	print("квантиль 1 - alpha / 2:", t_val)
	print("phi2 статистика:", phi2)
	if fabs(phi2) < t_val:
		print("принимаем гипотезу о равенстве матожиданий")
	else:
		print("отвергаем гипотезу о равенстве матожиданий")
	return 0

# модифицированная статистика Колмогорова для сложных гипотез
def Kolmogorov_stat(data, n, a, std):
	res = 0
	for i in range(n):
		f = norm.cdf(data[i], a, std)
		tmp = max((i + 1) / n - f, f - i / n)
		if tmp > res:
			res = tmp
	n_rt = sqrt(n)
	return res * (n_rt - 0.01 + 0.85 / n_rt)

def Kolmogorov_test(k1, k2):
	d = 0.895 # alpha = 0.05
	print("стат Колмогорова для набора 1: ", k1)
	print("стат Колмогорова для набора 2: ", k2)
	if k1 < d:
		print("принимаем гипотезу о норм распр 1")
	else:
		print("отвергаем гипотезу о норм распр 1")
	if k2 < d:
		print("принимаем гипотезу о норм распр 2")
	else:
		print("отвергаем гипотезу о норм распр 2")

# читаем данные
exp, n = read_file_data("data_5_1_exp")
imp, n = read_file_data("data_5_1_imp")

# вычисляем разности
n = n - 1
y1 = np.zeros(n)
y2 = np.zeros(n)

for i in range (n):
	y1[i] = exp[i + 1] - exp[i]
	y2[i] = imp[i + 1] - imp[i]

a1 = y1.mean()
a2 = y2.mean()
var1 = np.var(y1, ddof=1)
var2 = np.var(y2, ddof=1)

print("матожидание (экспорт):", a1)
print("дисперсия (экспорт):", var1)

print("матожидание (импорт):", a2)
print("дисперсия (импорт):", var2)


F_test(var1, var2, n, 0.05)
Student_test(a1, a2, var1, var2, n, 0.05)
k1 = Kolmogorov_stat(y1, n, a1, sqrt(var1))
k2 = Kolmogorov_stat(y2, n, a2, sqrt(var2))
Kolmogorov_test(k1, k2)




