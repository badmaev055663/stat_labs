from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, f, chi2, linregress

def read_file_data(filepath: str) -> tuple:
    file = open(filepath, "r")
    lines = file.readlines()
    n = int(len(lines) / 3)
    y = np.zeros(n)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    cnt = 0

    for line in lines:
        tmp = float(line.replace(',','.'))
        cnt += 1
        if cnt <= n:
            y[cnt - 1] = tmp
        elif cnt <= 2 * n:
            x1[cnt - n - 1] = tmp
        else:
            x2[cnt - 2 * n - 1] = tmp
    file.close()
    return (y, x1, x2)


def plot_regression(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, label: str):
    plt.scatter(x, y, color = "b")
    plt.plot(x, y_pred, color = "r")
    plt.suptitle('Парная регрессия ' + label)
  
    plt.xlabel('x')
    plt.ylabel('y')
  
    plt.show()

def plot_error(x: np.ndarray, e: np.ndarray, n: int, label: str):
    nums = np.arange(start=0, stop=n)
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle('Диаграммы остатков ' + label)
    plot1.set_ylabel('остатки')
    plot1.scatter(x, e, color = "r")
    plot1.set_xlabel('фактор')
    
    plot2.scatter(nums, e, color = "b")
    plot2.set_xlabel('номер наблюдений')
  

    plt.show()

# F-статистика для R2 - значимость модели парной регрессии
def validate_model(r2: float, n: int, alpha: float):
    f_stat = r2 / (1 - r2) * (n - 2)
    q = f.ppf(q=1-alpha, dfn=1, dfd=n-2)
    print("F-стат модели:", f_stat)
    print("f_crit:", q)
    if (f_stat > q):
        print("модель значима")
    else:
        print("модель НЕ значима!")


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

def pair_regression(x: np.ndarray, y: np.ndarray, alpha: float, label: str, k: float):
    res = linregress(x, y)
    n = np.size(x)
    b = res.slope
    a = res.intercept
    r2 = res.rvalue * res.rvalue
    b_err = res.stderr
    a_err = res.intercept_stderr
    print(".............\nпарная регрессия " + label)
    print("оценка b: ", b)
    print("оценка a: ", a)
    print("коэффициент детерминации:", r2)
    validate_model(r2, n, alpha)
    validate_ab(a, b, a_err, b_err, n, alpha)

    y_pred = a + b * x
    mean = np.mean(x)
    e = np.zeros(n)
    RSS = 0
    s = 0
    for i in range(n):
        e[i] = y[i] - y_pred[i]
        s += (x[i] - mean)**2
        RSS += e[i]**2

    xp = mean * k
    yp = xp * b + a
    m = sqrt(RSS / (n - 2)) * sqrt(1 + 1 / n + (xp - mean)**2 / s)
    t_stat = t.ppf(1-alpha/2, n-2)
    print("xp:", xp)
    print(f"интвервал точечного прогноза: {yp:.2f} +/- {t_stat * m:.2f}")
    plot_regression(x, y, y_pred, label=label)
    plot_error(x, e, n, label=label)
    Jarque_Bera_test(e, alpha)


def check_errors(y: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    n = np.size(y)
    e = np.zeros(n)
    RSS = 0
    y_m = np.mean(y)
    tmp = 0
    for i in range(n):
        e[i] = y[i] - y_pred[i]
        RSS += e[i]**2
        tmp += (y[i] - y_m)**2

    Jarque_Bera_test(e, alpha)
    R2 = 1 - RSS / tmp
    F = R2 * (n - 3) / (2 * (1 - R2))
    q = f.ppf(q=1-alpha, dfn=2, dfd=n-3)
    print("множ коэфф детерминации:", R2)
    print("значение критерия Фишера:", F)
    print("крит значение критерия:", q)
   
    S2 = RSS / (n - 3)
    nums = np.arange(start=0, stop=n)

    plt.suptitle('Диаграмма остатков (множ. регрессия)')
    plt.scatter(nums, e, color = "b")
    plt.xlabel('номер наблюдения')
    plt.ylabel('остатки')
    plt.show()
    return sqrt(S2)


def mult_regression(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, alpha: float, k: float):
    eps = 0.1
    n = np.size(y)
    dummy = np.ones(n)
    X_T = np.asarray([dummy, x1, x2])
    X = np.transpose(X_T)
    XTX = np.matmul(X_T, X)
    det = np.linalg.det(XTX)
    print(".............\nмножественная регрессия")
    if abs(det) > eps:
        XTX_inv = np.linalg.inv(XTX)
        tmp = np.matmul(XTX_inv, X_T)
        res = np.matmul(tmp, y)
        print("коэффициенты регрессии:", res)
    else:
        print("определитель близок 0:", det)
        return
    a = res[0]
    b1 = res[1]
    b2 = res[2]
    y_pred = b1 * x1 + b2 * x2 + a
    S = check_errors(y, y_pred, alpha=0.05)
    m_a = S * sqrt(XTX_inv[0][0])
    m_b1 = S * sqrt(XTX_inv[1][1])
    m_b2 = S * sqrt(XTX_inv[2][2])
    t_a = a / m_a
    t_b1 = b1 / m_b1
    t_b2 = b2 / m_b2
    print("t_a:", t_a)
    print("t_b1:", t_b1)
    print("t_b2:", t_b2)
    q = t.ppf(1-alpha/2, n-3)
    print("q (t):", q)
    x1p = np.mean(x1) * k
    x2p = np.mean(x2) * k
    X_p = np.asarray([1, x1p, x2p])
    tmp = np.matmul(X_p, XTX_inv)
    tmp2 = X_p.reshape((-1, 1))
    tmp = np.matmul(tmp, tmp2)
    
    yp = x1p * b1 + x2p * b2 + a
    m_y = S * sqrt(1 + tmp[0])
    print("x1p:", x1p, "x2p:", x2p)
    print("yp:", yp, "+-", m_y * q)


y, x1, x2 = read_file_data("data_2_5")
pair_regression(x1, y, alpha=0.05, label="y-x1", k=3.35)
pair_regression(x2, y, alpha=0.05, label="y-x2", k=3.35)
mult_regression(x1, x2, y, alpha=0.05, k=3.35)