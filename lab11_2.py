from math import cos, sin, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def generate_data(n: int, a: float, b1: float, b2: float, b3: float) -> tuple:
    eps = np.random.normal(loc=0, scale=1.5, size=n)
    y = np.zeros(n)
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    for i in range(n):
        t = i + 1
        f1[i] = a + b1 * t
        f2[i] = b2 * sin(2 * pi * t / 5) + b3 * cos(2 * pi * t / 5)
        y[i] = f1[i] + f2[i] + eps[i]
    return (f1, f2, y)


def plot_data(data1: np.ndarray, data2: np.ndarray, label: str, opt_t: np.ndarray=None, opt_y: np.ndarray=None):
    t = np.arange(np.size(data1)) + 1
    plt.suptitle(label)
    plt.plot(t, data1, color = "r")
    plt.ylabel('y')
    
    plt.plot(t, data2, color = "g")
    plt.xlabel('t')
    if np.size(opt_t) > 0:
        plt.scatter(opt_t, opt_y, color = "b")
    plt.show()

def plot_error(e: np.ndarray, n: int):
    t = np.arange(start=1, stop=n+1)
    plt.scatter(t, e, color = "r")
    plt.xlabel('t')
    plt.ylabel('остатки')
    plt.show()


def plot_simple(data: np.ndarray,  n: int, label: str):
    t = np.arange(start=1, stop=n+1)
    plt.title(label)
    plt.plot(t, data, color = "b")
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

def stats(y: np.ndarray, f1: np.ndarray, f2: np.ndarray, T: int):
    n = np.size(y)
    t = np.arange(n) + 1 
    reg = linregress(t, y)
    a = reg.intercept
    b = reg.slope
    tr = a + b * t
    plot_data(tr, f1, label="оценка тренда и истинный тренд")

    m = int(n / T)
    season = y - tr

    plot_simple(season, n, label="без тренда")
    S = np.zeros(T)
    j = 0
    for i in range(n):
        j = i % T
        S[j] += season[i]

    for j in range(T):
        S[j] /= m

    plot_data(S, f2[0:T], label="оценка сезонной компоненты и истинные значения")

    y_pred = tr
    for i in range(n):
        y_pred[i] += S[i % T]
    
   
    opt_t = np.array([31, 34, 37, 41])
    k = np.size(opt_t)
    opt_y = np.zeros(k)
    for i in range(k):
        t = opt_t[i]
        opt_y[i] = a + b * t + S[(t - 1) % T]

    plot_data(y, y_pred, label="исходный ряд и его расчетные значения", opt_t=opt_t, opt_y=opt_y)

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
    plot_error(e, n)

f1, f2, y = generate_data(n=30, a=3, b1=1.5, b2=4.5, b3=5.0)
  
stats(y, f1, f2, T=5)