from math import sqrt
import numpy as np
from scipy.stats import f

def read_file_data(filepath: str) -> list:
	file = open(filepath, "r")
	data = []
	for line in file.readlines():
		tokens = line.split()
		n = len(tokens)
		tmp = np.zeros(n)
		for i in range(n):
			tmp[i] = float(tokens[i].replace(',','.'))
		data.append(tmp)
	file.close()
	return data

def compute_means(data: list) -> tuple:
	means = []
	n = 0
	sum = 0
	for i in range(len(data)):
		means.append(np.mean(data[i]))
		n += len(data[i])
		sum += np.sum(data[i])

	return (means, sum / n, n)

def get_q_stats(data: list, means: list, total_mean: float) -> tuple:
	Q1 = 0
	Q2 = 0
	for i in range(len(data)):
		l = len(data[i])
		Q1 += l * (means[i] - total_mean)**2
		for j in range(l):
			Q2 += (means[i] - data[i][j])**2
	return (Q1, Q2)

data = read_file_data("data_8_1")

def check_means(data: list, alpha: float):
	means, total_mean, n = compute_means(data)
	print("means:", means)
	Q1, Q2 = get_q_stats(data, means, total_mean)
	print("Q1:", Q1)
	print("Q2:", Q2)
	k = len(data)
	stat = Q1 * (n - k) / (Q2 * (k - 1))
	print("stat:", stat)
	crit = f.ppf(q=1-alpha, dfn=k-1, dfd=n-k)
	print("crit:", crit)
	if stat > crit:
		print("принимаем альтернативную гипотезу: матожидания не равны")

def check_contrasts(data: list, alpha: float):
	means, total_mean, n = compute_means(data)
	k = len(data)
	n1 = len(data[0])
	n2 = len(data[1])
	n3 = len(data[2])
	_, Q2 = get_q_stats(data, means, total_mean)
	Lk1 = means[0] - means[1]
	Lk2 = means[0] - means[2]
	Lk3 = means[1] - means[2]
	Lk4 = (means[0] + means[2]) / 2 - means[1]
	print("Lk1:", Lk1)
	print("Lk2:", Lk2)
	print("Lk3:", Lk3)
	print("Lk4:", Lk4)
	q = f.ppf(q=1-alpha, dfn=k-1, dfd=n-k)

	s1 = Q2 / (n - k) * (1 / n1 + 1 / n2) * sqrt((k - 1) * q)
	s2 = Q2 / (n - k) * (1 / n1 + 1 / n3) * sqrt((k - 1) * q)
	s3 = Q2 / (n - k) * (1 / n2 + 1 / n3) * sqrt((k - 1) * q)
	s4 = Q2 / (n - k) * (0.25 / n1 + 0.25 / n3 + 1 / n2) * sqrt((k - 1) * q)
	l1 = [Lk1 - s1, Lk1 + s1]
	l2 = [Lk2 - s2, Lk2 + s2]
	l3 = [Lk3 - s3, Lk3 + s3]
	l4 = [Lk4 - s4, Lk4 + s4]
	print("l1:", l1)
	print("l2:", l2)
	print("l3:", l3)
	print("l4:", l4)

check_means(data, alpha=0.05)
check_contrasts(data, alpha=0.05)