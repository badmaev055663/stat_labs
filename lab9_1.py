import numpy as np
from scipy.stats import chi2

# возвращает таблицу и суммарные частоты по строкам, столбцам
def generate_table(n: int, r: int, s: int) -> tuple:
	a1, std1 = 50, 10
	a2, std2 = 10, 5
	x = np.random.normal(a1, std1, n)
	y = np.random.normal(a2, std2, n)
	A = np.histogram(x, bins=r)
	B = np.histogram(y, bins=s)
	table = np.zeros(shape=(r, s))
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

	return (table, A[0], B[0])

def test(table: np.ndarray, n_r: np.ndarray, m_c: np.ndarray, alpha: float):
	r = table.shape[0]
	s = table.shape[1]
	n = np.sum(table)
	X_sqr = 0
	Y_sqr = 0
	for i in range(r):
		for j in range(s):
			tmp = n_r[i] * m_c[j] / n
			X_sqr += (table[i][j] - tmp)**2 / tmp
			Y_sqr += 2 * table[i][j] * np.log(table[i][j] / tmp)
	
	print("X^2:", X_sqr)
	print("Y^2:", Y_sqr)
	q = chi2.ppf(1-alpha, (r - 1) * (s - 1))
	print("q:", q)
	if X_sqr < q:
		print("Принимаем нулевую гипотезу по критерию X^2")
	if Y_sqr < q:
		print("Принимаем нулевую гипотезу по критерию Y^2")
	
	

table, n_r, m_c = generate_table(n=100, r=3, s=5)
test(table, n_r, m_c, alpha=0.05)