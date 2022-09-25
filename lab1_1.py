import statistics
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew


# размер выборок, матожидание и отклонение
n1, mu1, sigma1 = 80, 30, 4
n2, mu2, sigma2 = 60, 20, 5

# генерация выборок
s1 = np.random.normal(mu1, sigma1, n1)
s2 = np.random.normal(mu2, sigma2, n2)

# объединенная и отсортированная выборка
s = np.concatenate([s1, s2])
s.sort()

# выборочные характеристики
n = len(s)
x_max = s[n - 1]
x_min = s[0]
mean = s.mean()
med = np.median(s)
mode = statistics.mode(s)
rng = x_max - x_min
var1 = np.std(s)
var2 = np.std(s, ddof=1)
mad = np.mean(np.absolute(s - mean))
kurt = kurtosis(s, fisher=True)
skw = skew(s)
cv = var1 / mean * 100
s_err = var1 / math.sqrt(n)

print("n:", n)
print("max:", x_max)
print("min:", x_min)
print("mean:", mean)
print("median:", med)
print("mode:", mode)
print("range:", rng)
print("dispersion biased:", var1)
print("dispersion unbiased:", var2)
print("std deviation:", math.sqrt(var1))
print("corrected std deviation:", math.sqrt(var2))
print("mean absolute deviation:", mad)
print("kurtosis:", kurt)
print("skew:", skw)
print("coefficient of variation:", cv, "%")
print("sample error:", s_err)


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


def show_hists(nbins):
    # гистограмма частот и гистограмма относительных частот
    hist1 = plt.subplot(2, 2, 1)
    plt.hist(s, edgecolor='black', bins=nbins)
    plt.title('Гистограмма частот')

    hist2 = plt.subplot(2, 2, 2)
    plt.hist(s, edgecolor='black', bins=nbins, weights=np.ones_like(s) / n)
    plt.title('Гистограмма относительных частот')

    plt.show()

    # гистограмма и полигон плотности относительных частот
    hist1 = plt.hist(s, edgecolor='black', bins=nbins, weights=np.ones_like(s) * nbins / (n * rng))
    add_freq_polygon(hist1, nbins, False)
    plt.title('Гистограмма и полигон плотности относительных частот')
    plt.show()

    # гистограмма и полигон относительных кумулятивных частот
    hist2 = plt.hist(s, edgecolor='black', weights=np.ones_like(s) / n, cumulative=True, bins=nbins)
    add_freq_polygon(hist2, nbins, True)
    plt.title('Гистограмма относительных кумулятивных частот')
    plt.show()

show_hists(10)

show_hists(8) # округление 3.2 * lgN + 1