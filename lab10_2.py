import numpy as np
from scipy.stats import f, linregress

def generate_data(n: int, a: float, b: float) -> tuple:
    eps = np.random.normal(loc=0, scale=1, size=n)
    x = np.random.normal(loc=10, scale=3, size=n)
    x = np.sort(x)
    y = np.zeros(n)
    for i in range(n):
        y[i] = a + b * x[i] + eps[i] * (1.06**i)
    return (x, y)


def get_stats(x: np.ndarray, y: np.ndarray) -> tuple:
    n = np.size(x)
    reg = linregress(x, y)
    a = reg.intercept
    b = reg.slope
    y_pred = a + b * x
    S = 0
    for i in range(n):
        S += (y[i] - y_pred[i])**2
    return (S, a, b)


def Golfeld_test(x: np.ndarray, y: np.ndarray, m: int, alpha: float):
    n = np.size(x)
    k = int((n - m) / 2)
    k2 = int((n + m) / 2)
    x1 = x[0: k + 1]
    y1 = y[0: k + 1]

    x2 = x[k2 - 1:]
    y2 = y[k2 - 1:]

    S1, a1, b1 = get_stats(x1, y1)
    S2, a2, b2 = get_stats(x2, y2)
    _, a, b = get_stats(x, y)
    print("исходная выборка")
    print(x)
    print("a:", a)
    print("b:", b)
    print(".........................\nвыборка 1")
    print(x1)
    print("S1:", S1)
    print("a1:", a1)
    print("b1:", b1)
    print(".........................\nвыборка 2")
    print(x2)
    print("S2:", S2)
    print("a2:", a2)
    print("b2:", b2)
    F_stat = S2 / S1
    print(".........................")
    print("F статистика:", F_stat)
    q = f.ppf(q=1-alpha, dfn=k-2, dfd=k-2)
    print("q:", q)

   

# https://studfile.net/preview/7431649/
np.set_printoptions(precision=2)
x, y = generate_data(n=30, a=10, b=3)
Golfeld_test(x, y, m=8, alpha=0.05)