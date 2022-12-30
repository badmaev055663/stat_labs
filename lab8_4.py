import numpy as np

def read_file_data(filepath: str) -> np.ndarray:
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
	return np.asarray(data)

def to_ranks(data: np.ndarray) -> np.ndarray:
	n = data.shape[0]
	k = data.shape[1]
	ranks = []

	for i in range(n):
		tmp = np.sort(data[i])
		row_ranks = []
		for j in range(k):
			for t in range(k):
				if data[i][j] == tmp[t]:
					row_ranks.append(t + 1)
		ranks.append(row_ranks)
	return np.asarray(ranks)

def Friedman_test(ranks: np.ndarray):
	n = ranks.shape[0]
	k = ranks.shape[1]

	# средние ранги по столбцам
	cranks = np.mean(ranks, axis=0)
	print(cranks)
	r = (k + 1) / 2
	stat = 0
	for j in range(k):
		stat += (r - cranks[j])**2
	
	stat *= 12 * n / (k * (k + 1))
	print("stat:", stat)
	crit = 7.0 # n = 3; k = 4; alpha = 0.05
	print("crit:", crit)
	if stat > crit:
		print("Отвергаем нулевую гипотезу")



data = read_file_data("data_8_3")
ranks = to_ranks(data)
Friedman_test(ranks)

