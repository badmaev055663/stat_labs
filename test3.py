from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, chi2

def read_file_data(filepath: str) -> np.ndarray:
    file = open(filepath, "r")
    lines = file.readlines()
    y = np.zeros(len(lines))
    cnt = 0

    for line in lines:
        y[cnt] = float(line.replace(',','.'))
        cnt += 1
    file.close()
    return y

def plot_data(data1: np.ndarray, data2: np.ndarray, label: str):
    t = np.arange(np.size(data1)) + 1
    plt.suptitle(label)
    plt.plot(t, data1, color = "r")
    plt.ylabel('продажи')
    
    plt.plot(t, data2, color = "g")
    plt.xlabel('t')
    plt.show()

def plot_error(e: np.ndarray, n: int):
    t = np.arange(start=1, stop=n+1)
    plt.scatter(t, e, color = "r")
    plt.xlabel('t')
    plt.ylabel('остатки')
    plt.show()


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

def stats(y: np.ndarray, T: int):
    n = np.size(y)
    t = np.arange(n) + 1 
    reg = linregress(t, y)
    a = reg.intercept
    b = reg.slope
    tr = a + b * t
    plot_data(tr, y, label="оценка тренда (кр.) и исходный ряд (зел.)")

    m = int(n / T)
    season = y - tr
    S = np.zeros(T)
    j = 0
    for i in range(n):
        j = i % T
        S[j] += season[i]

    for j in range(T):
        S[j] /= m

    y_pred = tr
    for i in range(n):
        y_pred[i] += S[i % T]
    
    t_2004 = n + 1
    k = 4
    y_2004 = np.zeros(k)
    for i in range(k):
        y_2004[i] = a + b * t_2004 + S[(t_2004 - 1) % T]
        t_2004 += 1

    print(y_2004)

    plot_data(y_pred, y, label="исходный ряд (зел.) и его расчетные значения (кр.)")

    e = np.zeros(n)
    DW = 0
    tmp = 0
    for i in range(n):
        e[i] = y[i] - y_pred[i]
        tmp += e[i]**2

    for i in range(n - 1):
        DW += (e[i + 1] -e[i])**2

    DW /= tmp
    print("DW:", DW)
    Jarque_Bera_test(e, alpha=0.05)
    plot_error(e, n)

y = read_file_data("data_3_20")
stats(y, T=4)