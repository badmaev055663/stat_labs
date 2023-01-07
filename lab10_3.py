from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, f, linregress, pearsonr

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
    print(".............\nпарная регрессия")
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


def check_errors(y: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    n = np.size(y)
    e = np.zeros(n)
    RSS = 0
    A = 0
    y_m = np.mean(y)
    tmp = 0
    for i in range(n):
        e[i] = y[i] - y_pred[i]
        A += abs(e[i] / y[i])
        RSS += e[i]**2
        tmp += (y[i] - y_m)**2

    R2 = 1 - RSS / tmp
    print("множ коэфф детерминации:", R2)
    F = R2 * (n - 3) / (2 * (1 - R2))
    q = f.ppf(q=1-alpha, dfn=2, dfd=n-3)
    print("значение критерия Фишера:", F)
    print("крит значение критерия:", q)
    A /= n
    print("Средняя ошибка аппроксимации:", A)

    S2 = RSS / (n - 3)
    nums = np.arange(start=0, stop=n)
 
    plt.scatter(nums, e, color = "b")
    plt.xlabel('n')
    plt.ylabel('error')
    plt.show()
    return sqrt(S2)

def mult_regression(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, alpha: float):
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
    y_pred = res[1] * x1 + res[2] * x2 + res[0]
    S = check_errors(y, y_pred, alpha=0.05)
    r_yx1 = pearsonr(y, x1)[0]
    r_yx2 = pearsonr(y, x2)[0]
    r_x1x2 = pearsonr(x1, x2)[0]
    r_x2x1 = pearsonr(x2, x1)[0]
    r1 = (r_yx1 - r_yx2 * r_x1x2) / sqrt((1 - r_yx2**2) * (1 - r_x1x2**2))
    r2 = (r_yx2 - r_yx1 * r_x2x1) / sqrt((1 - r_yx1**2) * (1 - r_x2x1**2))
    print("r1:", r1)
    print("r2:", r2)
    m_a = S * sqrt(XTX_inv[0][0])
    m_b1 = S * sqrt(XTX_inv[1][1])
    m_b2 = S * sqrt(XTX_inv[2][2])
    t_a = a / m_a
    t_b1 = b1 / m_b1
    t_b2 = b2 / m_b2
    print("m_a:", m_a)
    print("m_b1:", m_b1)
    print("m_b2:", m_b2)
    print("t_a:", t_a)
    print("t_b1:", t_b1)
    print("t_b2:", t_b2)
    q = t.ppf(1-alpha/2, n-3)
    print("q (t):", q)
    x1p = np.mean(x1) * 3.0
    x2p = np.mean(x2) * 3.0
    X_p = np.asarray([1, x1p, x2p])
    tmp = np.matmul(X_p, XTX_inv)
    tmp2 = X_p.reshape((-1, 1))
    tmp = np.matmul(tmp, tmp2)
    
    yp = x1p * b1 + x2p * b2 + a
    m_y = S * sqrt(1 + tmp[0])
    print("x1p:", x1p, "x2p:", x2p)
    print("yp:", yp, "+-", m_y * q)



x1, x2, y = generate_data(n=30, a=15, b1=3.0, b2=2.0)
np.set_printoptions(precision=3)
regression_stats(x1, y, alpha=0.05)
regression_stats(x2, y, alpha=0.05)
mult_regression(x1, x2, y, alpha=0.05)
