from math import cos, pi
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n: int) -> tuple:
    eps1 = np.random.normal(loc=0, scale=2.0, size=n)
    eps2 = np.random.normal(loc=0, scale=2.0, size=n)
    eps3 = np.random.normal(loc=0, scale=2.0, size=n)

    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)
    for i in range(n):
        t = i + 1
        y1[i] = 10 + 2.5 * t + 2.5 * cos(2 * pi * t / 5) + eps1[i]
        y2[i] = 15 + 2.0 * t + 2.0 * cos(2 * pi * t / 4) + eps2[i]
        y3[i] = 20 + 1.5 * t + 1.5 * cos(2 * pi * t / 3) + eps3[i]
    y3[20] = -500 + eps3[20]
    y3[29] = 500 + eps3[29]
    return (y1, y2, y3)

def plot_time_series(data: list, label: str):
    size = len(data1)
    
    plt.xlabel('t')
    plt.ylabel('Значения ряда y')
    labels = ["исх.", "g=3", "g=4", "g=5", "взвеш.", "мед."]
    shift = 1
    for i in range(size):
        t = np.arange(np.size(data[i])) + shift
        if i == 0 or i == 1:
            shift += 1
        plt.plot(t, data[i], label=labels[i])
    plt.title(label)
    plt.legend(loc='upper left')
 
    plt.show()

def plot_error(e1: np.ndarray, e2: np.ndarray, e3: np.ndarray, n: int, label: str):
    t = np.arange(start=1, stop=n+1)
    fig, (plot1, plot2, plot3) = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle('Остатки: ' + label)
    plot1.set_ylabel('Значения остатков')
    plot1.scatter(t, e1, color = "r")
    plot1.set_xlabel('ряд y1')
    
    plot2.scatter(t, e2, color = "g")
    plot2.set_xlabel('ряд y2')

    plot3.scatter(t, e3, color = "b")
    plot3.set_xlabel('ряд y3')
   
    plt.show()
 
def weighted_moving_average(data: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = np.size(data)
    m = np.size(w)
    ret = np.zeros(n - m)
    for i in range(n - m):
        tmp = data[i: i + m]
        ret[i] = np.dot(tmp, w)
    return ret

def moving_median(data: np.ndarray, m: int) -> np.ndarray:
    n = np.size(data)
    ret = np.zeros(n - m)
    for i in range(n - m):
        tmp = data[i: i + m]
        ret[i] = np.median(tmp)
    return ret


def get_errors(y: np.ndarray, y_: np.ndarray, n: int) -> np.ndarray:
    e = np.zeros(n)
    for i in range(n):
        e[i] = y[i] - y_[i]
    return e

av_w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
y1, y2, y3 = generate_data(40)
n = 30

y1_3 = weighted_moving_average(y1, w=np.ones(3) / 3)
y2_3 = weighted_moving_average(y2, w=np.ones(3) / 3)
y3_3 = weighted_moving_average(y3, w=np.ones(3) / 3)

e1_3 = get_errors(y1, y1_3, n)
e2_3 = get_errors(y2, y2_3, n)
e3_3 = get_errors(y3, y3_3, n)

y1_4 = weighted_moving_average(y1, w=np.ones(4) / 4)
y2_4 = weighted_moving_average(y2, w=np.ones(4) / 4)
y3_4 = weighted_moving_average(y3, w=np.ones(4) / 4)

e1_4 = get_errors(y1, y1_4, n)
e2_4 = get_errors(y2, y2_4, n)
e3_4 = get_errors(y3, y3_4, n)

y1_5 = weighted_moving_average(y1, w=np.ones(5) / 5)
y2_5 = weighted_moving_average(y2, w=np.ones(5) / 5)
y3_5 = weighted_moving_average(y3, w=np.ones(5) / 5)

e1_5 = get_errors(y1, y1_5, n)
e2_5 = get_errors(y2, y2_5, n)
e3_5 = get_errors(y3, y3_5, n)

w = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

y1_w = weighted_moving_average(y1, w)
y2_w = weighted_moving_average(y2, w)
y3_w = weighted_moving_average(y3, w)

e1_w = get_errors(y1, y1_w, n)
e2_w = get_errors(y2, y2_w, n)
e3_w = get_errors(y3, y3_w, n)

y1_m = moving_median(y1, 4)
y2_m = moving_median(y2, 4)
y3_m = moving_median(y3, 4)

e1_m = get_errors(y1, y1_m, n)
e2_m = get_errors(y2, y2_m, n)
e3_m = get_errors(y3, y3_m, n)


data1 = [y1, y1_3, y1_4, y1_5, y1_w, y1_m]
plot_time_series(data1, label="y1")

data2 = [y2, y2_3, y2_4, y2_5, y2_w, y2_m]
plot_time_series(data2, label="y2")

data3 = [y3, y3_3, y3_4, y3_5, y3_w, y3_m]
plot_time_series(data3, label="y3")

plot_error(e1_3, e2_3, e3_3, n, label="сглаживание g=3")
plot_error(e1_4, e2_4, e3_4, n, label="сглаживание g=4")
plot_error(e1_5, e2_5, e3_5, n, label="сглаживание g=5")
plot_error(e1_w, e2_w, e3_w, n, label="сглаживание взвеш g=5")
plot_error(e1_m, e2_m, e3_m, n, label="сглаживание медианное g=4")
