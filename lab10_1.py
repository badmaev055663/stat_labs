from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, f, linregress

def generate_data(n: int, a: float, b: float) -> tuple:
    eps = np.random.normal(loc=0, scale=2, size=n)
    x = np.random.normal(loc=10, scale=3, size=n)
    y = np.zeros(n)
    for i in range(n):
        y[i] = a + b * x[i] + eps[i]
    return (x, y)

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

# F-статистика для R2 - значимость модели регрессии
def validate_model(r2: float, n: int, alpha: float):
    f_stat = r2 / (1 - r2) * (n - 2)
    q = f.ppf(q=1-alpha, dfn=1, dfd=n-2)
    print("f_stat:", f_stat)
    print("f_crit:", q)
    if (f_stat > q):
        print("модель значима")
    else:
        print("модель НЕ значима!")

# t-статистика для корреляции
def validate_cor(r: float, n: int, alpha: float):
    t_crit = t.ppf(1-alpha/2, n-2)
    t_stat = r * sqrt(n - 2)/ sqrt(1 - r * r) 
    print("t:", t_stat)
    print("t_crit:", t_crit)
    if (abs(t_stat) > t_crit):
        print("корреляция значима")
    else:
        print("нет корреляции!")

# t-статистика и интевалы для a и b
def validate_ab(a: float, b: float, a_err: float, b_err: float, n: int, alpha: float):
    t_a = a / a_err
    t_b = b / b_err
    print("t a: ", t_a)
    print("t b: ", t_b)
    t_stat = t.ppf(1-alpha/2, n-2)
    print("t_stat:", t_stat)
    if t_a > t_stat and t_b > t_stat:
        print("коэффициенты регрессии значимы")
    else:
        print("коэффициенты регрессии НЕ значимы!")
    print(f"интервал b: {b:.4f} +/- {t_stat * b_err:.4f}")
    print(f"интервал a: {a:.4f}" f" +/- {t_stat * a_err:.4f}")


def regression_stats(x: np.ndarray, y: np.ndarray, k: float, alpha: float):
    res = linregress(x, y)
    n = np.size(x)
    b = res.slope
    a = res.intercept
    r = res.rvalue
    r2 = r * r
    b_err = res.stderr
    a_err = res.intercept_stderr
    print("b: ", b)
    print("a: ", a)
    print("b std err: ", b_err)
    print("a std err: ", a_err)
    print("коэффициент корреляции: ", r)
    print("коэффициент детерминации:", r2)
    validate_cor(r, n, alpha)
    validate_model(r2, n, alpha)

    y_pred = a + b * x
    plot_regression(x, y, y_pred)
    mean = np.mean(x)
    e = np.zeros(n)
    A = 0
    RSS = 0
    s = 0
    for i in range(n):
        e[i] = y[i] - y_pred[i]
        A += abs(e[i] / y[i])
        RSS += e[i]**2
        s += (x[i] - mean)**2
    A /= n
    xp = mean * k
    yp = xp * b + a
    m = sqrt(RSS / (n - 2)) * sqrt(1 + 1 / n + (xp - mean)**2 / s)
    t_stat = t.ppf(1-alpha/2, n-2)
    print(f"интвервал точечного прогноза: {yp:.2f} +/- {t_stat * m:.2f}")
    print("xp:", xp)
    print("средняя ошибка аппроксимации: ", A)

    validate_ab(a, b, a_err, b_err, n, alpha)
    plot_regression(x, y, y_pred)
    plot_error(x, e, n)

x, y = generate_data(n=50, a=10, b=2)
regression_stats(x, y, k=3, alpha=0.05)
