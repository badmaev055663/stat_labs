import numpy as np
from scipy.stats import chi2

def read_file_data(filepath: str) -> list:
	file = open(filepath, "r")
	data = []
	for line in file.readlines():
		tokens = line.split()
		n = len(tokens)
		tmp = np.zeros(n)
		for i in range(n):
			tmp[i] = float(tokens[i].replace(',','.'))
		data.append(np.sort(tmp))
	file.close()
	return data


def process_data(data: list) -> np.ndarray:
	data_plain = np.array([])
	for i in range(len(data)):
		data_plain = np.append(data_plain, data[i])

	data_plain = np.sort(data_plain)
	return data_plain

# с учетом совпадающих значений
def average_rank(row: np.ndarray, data_plain: np.ndarray) -> float:
	total = 0
	j = 0
	l = len(row)
	n = len(data_plain)
	for i in range(n):
		if j == l:
			break
		if row[j] == data_plain[i]:
			if i < n - 1 and row[j] == data_plain[i + 1]:
				total += (i + 1.5)
			else:
				total += (i + 1)
			j += 1
	return total / l

def H_test(data: list, alpha: float):
	data_plain = process_data(data)
	rank1 = average_rank(data[0], data_plain)
	rank2 = average_rank(data[1], data_plain)
	rank3 = average_rank(data[2], data_plain)
	n = len(data_plain)
	k = len(data)
	n1 = len(data[0])
	n2 = len(data[1])
	n3 = len(data[2])
	H = n1 * (rank1 - (n + 1) / 2)**2 + n2 * (rank2 - (n + 1) / 2)**2 + n3 * (rank3 - (n + 1) / 2)**2 
	H = H * 12 / (n * (n + 1))
	q = chi2.ppf(1-alpha, k - 1)
	print("q:", q)
	print("H:", H)
	if H > q:
		print("выборки неоднородны")



data = read_file_data("data_8_1")
H_test(data, alpha=0.05)

