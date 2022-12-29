import numpy as np
from scipy.stats import chi2, f

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

def test(data: np.ndarray, alpha: float):
	m = np.mean(data)
	m_r = np.mean(data, axis=1)
	m_c = np.mean(data, axis=0)
	k = data.shape[1]
	n = data.shape[0]
	SR = SG = S1 = 0
	for j in range(k):
		SG += (m_c[j] - m)**2
		for i in range(n):
			S1 += (data[i][j] - m_r[i] - m_c[j] + m)**2
	
	for i in range(n):
		SR += (m_r[i] - m)**2

	SR *= k
	SG *= n

	F1 = SR * (k - 1) / S1
	q1 = f.ppf(q=1-alpha, dfn=n-1, dfd=(n-1)*(k-1))
	print("F1:", F1)
	print("q1:", q1)
	if F1 > q1:
		print("отвергаем первую нулевую гипотезу: есть эффект по строкам")

	F2 = SG * (n - 1) / S1
	q2 = f.ppf(q=1-alpha, dfn=k-1, dfd=(n-1)*(k-1))
	print("F2:", F2)
	print("q2:", q2)
	if F2 > q2:
		print("отвергаем вторую нулевую гипотезу: есть эффект по столбцам")



data = read_file_data("data_8_3")
test(data, alpha=0.05)