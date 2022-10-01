import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli

# добавить полигон частот (к гистограмме)
def add_freq_polygon(hist, nbins, is_cumulative):
    x = np.empty(nbins)
    tmp = np.zeros(1)
    tmp[0] = hist[1][0]
    for i in range(nbins):
        x[i] = (hist[1][i] + hist[1][i + 1]) / 2
    x = np.append(tmp, x)
    x = np.append(x, hist[1][nbins])
    y = np.zeros(1)
    y = np.append(y, hist[0])
    if is_cumulative == True:
        y = np.append(y, np.ones(1))
    else:
        y = np.append(y, np.zeros(1))
    plt.plot(x, y)


def show_hist_freq(s, nbins1, nbins2, n):
    hist1 = plt.subplot(2, 2, 1)
    hist1 = plt.hist(s, edgecolor='black', bins=nbins1, weights=np.ones_like(s) / n)
    add_freq_polygon(hist1, nbins1, False)
    plt.title('Гистограмма и полигон 1')

    hist2 = plt.subplot(2, 2, 2)
    hist2 = plt.hist(s, edgecolor='black', bins=nbins2, weights=np.ones_like(s) / n)
    add_freq_polygon(hist2, nbins2, False)
    
    plt.title('Гистограмма и полигон 2')
    plt.show()

def show_hist(s, nbins1, nbins2, n):
    hist1 = plt.subplot(2, 2, 1)
    plt.hist(s, edgecolor='black', bins=nbins1, weights=np.ones_like(s) / n)
    
    plt.title('Гистограмма 1')

    hist2 = plt.subplot(2, 2, 2)
    plt.hist(s, edgecolor='black', bins=nbins2, weights=np.ones_like(s) / n)
 
    plt.title('Гистограмма 2')
    plt.show()   

def get_interval_p(a, std, l1, l2):
    p1 = norm(loc=a, scale=std).cdf(l1)
    p2 = norm(loc=a, scale=std).cdf(l2)
    return p2 - p1

# дов интервал при известном значении станд отклонения
def mean_conf_interval(s, eps, std):
    mean = s.mean()
    n = len(s)
    z = norm.ppf(1 - eps / 2, loc=0, scale=1)
    interval = (mean - z * std / math.sqrt(n), mean + z * std / math.sqrt(n))
    print("интервал:", interval)
    print("проверка:", norm.interval(1 - eps, loc=mean, scale=std/math.sqrt(n)))
  

# дов интервал по выборочному станд отклонению 
def mean_conf_interval2(mean, std, n):
    # захардкодили t (Стьюдента)
    if n == 200:
        t = 1.972
    else:
        t = 2.045
    interval = (mean - t * std / math.sqrt(n), mean + t * std / math.sqrt(n))
    print("интервал (выб):", interval)


# дов интервал при известном значении матожидания
def var_std_conf_interval(std, n):
    # захардкодили t (Пирсона хи-квадрат)
    if n == 200:
        t1 = 239.9
        t2 = 161.8
    else:
        t1 = 45.7
        t2 = 16.0
    var_interval = (std**2 * (n - 1) / t1, std**2 * (n - 1) / t2)
    std_interval = (std * math.sqrt((n - 1) / t1), std * math.sqrt((n - 1) / t2))
    print("интервал для дисперсии (выб):", var_interval)
    print("интервал для станд отклонения (выб):", std_interval)


def test():
    # размер выборки, матожидание и отклонение
    n, mu, sigma = 200, 15, 5
    # границы интервала
    l1, l2 = 6, 11
    
    # 1. выборочные характеристики и генральная совокупность
    s = np.random.normal(mu, sigma, n)

    show_hist_freq(s, 8, 12, n)
 

    std = np.std(s, ddof=0)
    a = s.mean()
   
    print("std deviation", std)
    print("mean", a)

    # 2. вероятности попадания в интервал
    p1 = get_interval_p(a, std, l1, l2)
    p2 = get_interval_p(mu, sigma, l1, l2)
    print("вероятность попадания для исходной совокупности:", p2)
    print("вероятность попадания для сгенерированной совокупности:", p1)
    print("------------------------------------------------")

    # 3. выборки размера 30 и их гистограммы и прочее
    m = 30
    s1 = np.random.normal(mu, sigma, m)
    s2 = np.random.normal(mu, sigma, m)

    a1 = s1.mean()
    a2 = s2.mean()
    std1 = np.std(s1, ddof=1)
    std2 = np.std(s2, ddof=1)
    show_hist(s1, 5, 7, m)
    show_hist(s2, 5, 7, m)
   

    eps = 0.05

    # 4. доверительные интервалы для матожидания с проверкой для 3 выборок
    # при известном значении стандартного отклонения
    mean_conf_interval(s, eps, sigma)
    mean_conf_interval(s1, eps, sigma)
    mean_conf_interval(s2, eps, sigma)
    print("------------------------------------------------")

    # 5. доверительные интервалы для матожидания для 3 выборок
    # при неизвестном значении генерального стандартного отклонения
    mean_conf_interval2(a, np.std(s, ddof=1), n)
    mean_conf_interval2(a1, std1, m)
    mean_conf_interval2(a2, std2, m)
    print("------------------------------------------------")

    # 6. доверительные интервалы для дисперсии и станд отклонения для 3 выборок
    # при неизвестном значении генерального матожидания
    var_std_conf_interval(np.std(s, ddof=1), n)
    var_std_conf_interval(std1, m)
    var_std_conf_interval(std2, m)
    print("------------------------------------------------")

    # 7. Моделируем распределение Бернулли
    k = 500
    p = 0.7 
    data_bern = bernoulli.rvs(size=k, p=p)
    u = norm.ppf(1 - eps / 2, loc=0, scale=1)
    x_m = data_bern.mean()
    delta = u * math.sqrt(x_m * (1 - x_m) / k)
    print(x_m)
    interval = (x_m - delta, x_m + delta)
    print(interval)

test()