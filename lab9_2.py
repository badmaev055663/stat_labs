import numpy as np
from math import factorial

def read_file_data(filepath: str) -> np.ndarray:
	file = open(filepath, "r")
	pacient = []
	dlt = []
	N = []
	remiss = []
	cnt = 0
	for line in file.readlines():
		tmp = float(line.replace(',','.'))
		if cnt % 4 == 0:
			pacient.append(tmp)
		elif cnt % 4 == 1:
			dlt.append(tmp)
		elif cnt % 4 == 2:
			N.append(tmp)
		elif cnt % 4 == 3:
			remiss.append(tmp)
		cnt += 1
	file.close()
	data = [pacient, dlt, N, remiss]
	return np.asarray(data, dtype=np.int64)


def build_table(x: np.ndarray, y: np.ndarray, r: int, s: int) -> np.ndarray:
	A = np.histogram(x, bins=r)
	B = np.histogram(y, bins=s)
	n = len(x)
	table = np.zeros(shape=(r, s), dtype=np.int64)
	i = j = 0
	for k in range(n):
		for t in range(r):
			if x[k] <= A[1][t + 1]:
				i = t
				break			
		for t in range(s):
			if y[k] <= B[1][t + 1]:
				j = t
				break
		table[i][j] += 1

	return table

def p_value(table: np.ndarray) -> float:
	n1f = factorial(table[0][0] + table[0][1])
	n2f = factorial(table[1][0] + table[1][1])
	m1f = factorial(table[0][0] + table[1][0])
	m2f = factorial(table[0][1] + table[1][1])
	nf = factorial(np.sum(table))
	a = factorial(table[0][0])
	b = factorial(table[0][1])
	c = factorial(table[1][0])
	d = factorial(table[1][1])

	p = n1f * n2f * m1f * m2f / (nf * a * b *c * d)
	return p

def test(data: np.ndarray, alpha: float):
	table1 = build_table(data[1], data[3], r=2, s=2)
	table2 = build_table(data[2], data[3], r=2, s=2)
	print("table1")
	print(table1)
	print("table1")
	print(table2)
	p1 = p_value(table1)
	p2 = p_value(table2)
	print("p1:", p1)
	print("p2:", p2)
	if p1 > alpha:
		print("нет связи между ДЛТ и ремиссией")
	if p2 < alpha:
		print("есть связь между N и ремиссией")


data = read_file_data("data_9_2")
test(data, alpha=0.05)

