import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f, linregress

def generate_data(n: int, a: float, b1: float, b2: float) -> tuple:
    eps = np.random.normal(loc=0, scale=2, size=n)
    x1 = np.random.normal(loc=10, scale=3, size=n)
    x2 = np.random.normal(loc=20, scale=4, size=n)
    y = np.zeros(n)
    for i in range(n):
        y[i] = a + b1 * x1[i] + b2 * x2[i] + eps[i]
    return (x1, x2, y)

def plot_regression(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
    plt.scatter(x, y, color = "b")
    plt.plot(x, y_pred, color = "r")
  
    plt.xlabel('x')
    plt.ylabel('y')
  
    plt.show()

def plot_error(x: np.ndarray, e: np.ndarray, n: int):
    nums = np.arange(start=0, stop=n)
    plt.scatter(x, e, color = "r")
  
    plt.xlabel('x')
    plt.ylabel('error')
    plt.show()
    plt.scatter(nums, e, color = "b")
  
    plt.xlabel('n')
    plt.ylabel('error')
    plt.show()

# F-статистика для R2 - значимость модели парной регрессии
def validate_model(r2: float, n: int, alpha: float):
    f_stat = r2 / (1 - r2) * (n - 2)
    q = f.ppf(q=1-alpha, dfn=1, dfd=n-2)
    print("f_stat:", f_stat)
    print("f_crit:", q)
    if (f_stat > q):
        print("модель значима")
    else:
        print("модель НЕ значима!")

def regression_stats(x: np.ndarray, y: np.ndarray, alpha: float):
    res = linregress(x, y)
    n = np.size(x)
    b = res.slope
    a = res.intercept
    r = res.rvalue
    r2 = r * r
    print("b: ", b)
    print("a: ", a)
    print("коэффициент корреляции: ", r)
    print("коэффициент детерминации:", r2)
    validate_model(r2, n, alpha)

    y_pred = a + b * x
    plot_regression(x, y, y_pred)
    e = np.zeros(n)
    A = 0
    for i in range(n):
        e[i] = y[i] - y_pred[i]
        A += abs(e[i] / y[i])
    A /= n
    print("средняя ошибка аппроксимации: ", A)

    plot_regression(x, y, y_pred)
    plot_error(x, e, n)

x1, x2, y = generate_data(n=30, a=15, b1=2, b2=2.5)
