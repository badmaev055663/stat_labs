from math import sqrt
import numpy as np
from scipy.stats import norm, chi2, t

def t_stat(data, n, a):
	mu = np.mean(data)
	s = np.std(data, ddof=1)
	return (mu - a) / s * sqrt(n)


# гипотезы о матожидании для норм распр
def test1(std, a0, a1, alpha, beta):
	z_a = norm.ppf(1-alpha)
	z_b = norm.ppf(beta)
	n = int(std * std * (z_a - z_b)**2 / (a1 - a0)**2) + 1
	c1 = a0 - z_a * std / sqrt(n)
	print("c1:", c1)
	print("n:", n)
	s0 = np.random.normal(a0, std, n)
	s1 = np.random.normal(a1, std, n)

	mu0 = np.mean(s0)
	mu1 = np.mean(s1)
	print("mu0:", mu0)
	print("mu1:", mu1)
	if mu0 > c1:
		print("для выборки 0 принимаем гипотезу a=a0")
	if mu1 <= c1:
		print("для выборки 1 отвергаем гипотезу a=a0")

	# сложная двусторонняя альтернатива при известном ст отклонении
	z = norm.ppf(1 - alpha/2)
	c2 = z * std / sqrt(n)
	print("границы крит областей: ", (a0 - c2), ";", (a0 + c2))
	if mu0 > (a0 - c2) and mu0 < (a0 + c2):
		print("для выборки 0 принимаем гипотезу a=a0 со сложной двусторонней альтернативой")
	if mu1 < (a0 - c2) or mu1 > (a0 + c2):
		print("для выборки 1 отвергаем гипотезу a=a0 со сложной двусторонней альтернативой")

	# сложная двусторонняя альтернатива при неизвестном ст отклонении
	t0 = t_stat(s0, n, a0)
	t1 = t_stat(s1, n, a0)
	print("t0", t0)
	print("t1", t1)
	q1 = t.ppf(alpha / 2, n - 1)
	q2 = t.ppf(1 - alpha / 2, n - 1)
	print("границы крит областей: ", q1, ";", q2)
	if t0 > q1 and t0 < q2:
		print("для выборки 0 принимаем гипотезу a=a0 со сложной двусторонней альтернативой при неизестном std")
	if t1 < q1 or t1 > q2:
		print("для выборки 1 отвергаем гипотезу a=a0 со сложной двусторонней альтернативой при неизестном std")

# гипотеза о станд отклонении для норм распр
def test2(std0, std1, a, n, alpha):
	s0 = np.random.normal(a, std0, n)
	s1 = np.random.normal(a, std1, n)
	stat0 = np.var(s0, ddof=0) * n
	stat1 = np.var(s1, ddof=0) * n
	print("stat0:", stat0)
	print("stat1:", stat1)
	c1 = std1 * std1 * chi2.ppf(1-alpha, n)
	print("c1:", c1)
	if stat0 > c1:
		print("для выборки 0 принимаем гипотезу sigma=sigma0")
	if stat1 <= c1:
		print("для выборки 1 отвергаем гипотезу sigma=sigma0")

# простая гипотеза о p для бином распр
def test3(p0, p1, n, alpha):
	s0 = np.random.binomial(n, p0)
	s1 = np.random.binomial(n, p1)
	m0 = np.sum(s0)
	m1 = np.sum(s1)
	c1 = n * p0 + sqrt(n * p0 * (1 - p0)) * norm.ppf(1-alpha)
	print("c1:", c1)
	print("m0:", m0)
	print("m1:", m1)
	if m0 > c1:
		print("для выборки 0 отвергаем гипотезу p=p0")
	else:
		print("для выборки 0 принимаем гипотезу p=p0")
	if m1 > c1:
		print("для выборки 1 отвергаем гипотезу p=p0")
	
test1(std=5, a0=10, a1=9, alpha=0.05, beta=0.1)
test2(std0=4, std1=3, a=5, n=50, alpha=0.05)
test3(p0=0.3, p1=0.35, n=500, alpha=0.05)