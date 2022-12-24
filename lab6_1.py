import numpy as np
from scipy.stats import binom

def read_file_data(filepath):
	file = open(filepath, "r")
	lines = file.readlines()
	n = len(lines)
	data = np.zeros(n)
	for i in range(n):
		data[i] = float(lines[i].replace(',','.'))
	file.close()
	return data, n

# читаем данные и вычисляем разности
data, n = read_file_data("data_6_1")
n = n - 1
x = np.zeros(n)

for i in range (n):
	x[i] = data[i + 1] - data[i]

def S_stat(x, n, theta):
        s = 0
        for i in range (n):
                if (x[i] - theta) > 0:
                        s += 1
        return s

def sign_test(x, n, theta, alpha):
        s = S_stat(x, n, theta)
        q1 = binom.ppf(alpha / 2, n, 0.5)
        q2 = binom.ppf(1 - alpha / 2, n, 0.5)
        print("S:", s)
        print("q1:", q1)
        print("q2:", q2)
        if s > q1 and s < q2:
                print('принимаем гипотезу')
        else:
                print('отвергаем гипотезу')


sign_test(x, n, 1.5, 0.05)









