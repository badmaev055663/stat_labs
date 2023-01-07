from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, f, chi2, linregress

def read_file_data(filepath: str) -> list:
    file = open(filepath, "r")
    lines = file.readlines()
    n = int(len(lines) / 3)
    y = np.zeros(n)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    cnt = 0

    for line in lines:
        tmp = float(line)
        cnt += 1
        if cnt <= n:
            y[cnt - 1] = tmp
        elif cnt <= 2 * n:
            x1[cnt - n - 1] = tmp
        else:
            x2[cnt - 2 * n - 1] = tmp
    file.close()
    return [y, x1, x2]

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


def regression_stats(x: np.ndarray, y: np.ndarray, k: float, alpha: float) -> np.ndarray:
    res = linregress(x, y)
    n = np.size(x)
    b = res.slope
    a = res.intercept
    r = res.rvalue
    r2 = r * r
    b_err = res.stderr
    a_err = res.intercept_stderr
    print("оценка b: ", b)
    print("оценка a: ", a)
    print("ст ошибка b: ", b_err)
    print("ст ошибка a: ", a_err)
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
    return e

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

def Jarque_Bera_test(e: np.ndarray, alpha: float):
    n = np.size(e)
    sum3 = 0
    sum4 = 0
    sigma2 = 0
    for i in range(n):
        sigma2 += e[i]**2
        sum3 += e[i]**3
        sum4 += e[i]**4
    sigma2 /= n
    S = sum3 / (n * sigma2**1.5)
    K = sum4 / (n * sigma2**2)
    JB = S**2 / 6 + (K - 3)**2 / 24
    print("JB:", JB)
    q = chi2.ppf(1-alpha, 2)
    print("q (JB):", q)
    if (q > JB):
        print("считаем ошибки нормально распределенными")
    else:
        print("ошибки НЕ распределены нормально!")


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
    F_stat = S2 / S1
    print(".........................")
    print("F статистика в тесте Гольдфельда:", F_stat)
    q = f.ppf(q=1-alpha, dfn=k-2, dfd=k-2)
    print("крит значение q:", q)
    if F_stat >= q:
        print("отвергаем гипотезу гомоскедастичности")
    else:
        print("принимаем гипотезу гомоскедастичности")


y, x1, x2 = read_file_data("data_1_11")

print("эффективность и IQ")
e1 = regression_stats(x=x1, y=y, k=1.7, alpha=0.05)
print("\n.....................\n")
Jarque_Bera_test(e1, alpha=0.05)
Golfeld_test(x=x1, y=y, m=8, alpha=0.05)
print("\n.....................\n")
print("эффективность и время работы")
e2 = regression_stats(x=x2, y=y, k=1.7, alpha=0.05)
Jarque_Bera_test(e2, alpha=0.05)
Golfeld_test(x=x2, y=y, m=8, alpha=0.05)