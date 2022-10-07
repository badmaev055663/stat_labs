import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def read_file_data(filepath):
	file = open(filepath, "r")
	lines = file.readlines()
	n = len(lines)
	years = np.zeros(n)
	prices = np.zeros(n)
	for i in range(n):
		years[i] = 1970 + i
		prices[i] = float(lines[i].replace(',','.'))
	file.close()
	return years, prices, n


def Kolmogorov_stat(data, n, a, std):
	res = 0
	for i in range(n):
		f = norm.cdf(data[i], a, std)
		tmp = max((i + 1) / n - f, f - i / n)
		if tmp > res:
			res = tmp
	n_rt = math.sqrt(n)
	return res * (n_rt - 0.01 + 0.85 / n_rt)

def w_sqr_stat(data, n, a, std):
	res = 1 / (12 * n)
	for i in range(n):
		f = norm.cdf(data[i], a, std)
		tmp = (2 * i + 1) / (2 * n)
		res += (f - tmp)*(f - tmp)
	
	return res * (1 + 0.5 / n)

# вычисляем разности и сортируем
years, prices, n = read_file_data("data_3_1")
y = np.zeros(n - 1)
n = n - 1
for i in range (n):
	y[i] = prices[i + 1] - prices[i]
y.sort()

# чтобы все значения были различны
y[13] = 0.071

# гистограмма распределения разностей
hist2 = plt.hist(y, edgecolor='black', weights=np.ones_like(y) / n, bins=6)
plt.show()

std = np.std(y, ddof=1)
print(std)
a = y.mean()
print(a)

q1_k = 0.895 # alpha = 0.05
q2_k = 0.819 # alpha = 0.1

q1_w = 0.126 # alpha = 0.05
q2_w = 0.104 # alpha = 0.1

stat1 = Kolmogorov_stat(y, n, a, std)
stat2 = w_sqr_stat(y, n, a, std)
print(stat1)
print(stat2)

