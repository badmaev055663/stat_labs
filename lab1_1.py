import statistics
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm, pearsonr

# выборочные характеристики 
def get_descr_stat(s):
    n = len(s)
    x_max = s[n - 1]
    x_min = s[0]
    mean = s.mean()
    med = np.median(s)
    mode = statistics.mode(s)
    rng = x_max - x_min
    var1 = np.var(s)
    var2 = np.var(s, ddof=1)
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


def show_hists(s, nbins, n):
    # гистограмма частот и гистограмма относительных частот
    hist1 = plt.subplot(2, 2, 1)
    plt.hist(s, edgecolor='black', bins=nbins)
    plt.title('Гистограмма частот')

    hist2 = plt.subplot(2, 2, 2)
    plt.hist(s, edgecolor='black', bins=nbins, weights=np.ones_like(s) / n)
    plt.title('Гистограмма относительных частот')

    plt.show()

    # гистограмма и полигон плотности относительных частот
    hist1 = plt.hist(s, edgecolor='black', bins=nbins, weights=np.ones_like(s) * nbins / (s[n - 1] - s[0]))
    add_freq_polygon(hist1, nbins, False)
    plt.title('Гистограмма и полигон плотности относительных частот')
    plt.show()

    # гистограмма и полигон относительных кумулятивных частот
    hist2 = plt.hist(s, edgecolor='black', weights=np.ones_like(s) / n, cumulative=True, bins=nbins)
    add_freq_polygon(hist2, nbins, True)
    plt.title('Гистограмма относительных кумулятивных частот')
    plt.show()


def descr_stat_test():
    # размер выборок, матожидание и отклонение
    n1, mu1, sigma1 = 80, 30, 4
    n2, mu2, sigma2 = 60, 20, 5
    n = n1 + n2
    # генерация выборок
    s1 = np.random.normal(mu1, sigma1, n1)
    s2 = np.random.normal(mu2, sigma2, n2)

    # объединенная и отсортированная выборка
    s = np.concatenate([s1, s2])
    s.sort()

    get_descr_stat(s)
    show_hists(s, 10, n)
    show_hists(s, 8, n) # округление 3.2 * lgN + 1

def validate(r, n):
    # критическое значение для p=0.05 и n ~ 140
    t_crit = 1.977
    t = r * math.sqrt(n - 2)/ math.sqrt(1 - r * r) 
    print("correlation:", r)
    print("t:", t)
    if (math.fabs(t) > t_crit):
        print("correlation is significant")
    else:
        print("no correlation")
    return 

def get_r_from_cov(s1, s2):
    cov = np.cov(s1, s2)[0][1]
    d1 = np.std(s1, ddof=1)
    d2 = np.std(s2, ddof=1)
    r = cov / (d1 * d2)
    print("covariation", cov)
    print("correlation from covariation", r)


def correlation_test():
    # размер выборок, матожидание и отклонение
    n = 140
    mu1, sigma1 = 30, 4
    mu2, sigma2 = 20, 5

    # равномерно распределенные случайные числа
    s = np.random.uniform(0, 1, n)

    # нормально распределенная выборка 1 полученная по случайным числам 
    s1 = np.zeros(n)
    for i in range(n):
        s1[i] = norm.ppf(s[i], loc=mu1, scale=sigma1)

    # нормально распределенная выборка 2 
    s2 = np.random.normal(mu2, sigma2, n)
    plt.subplot(2, 2, 1)
    plt.hist(s1, edgecolor='black', bins=8)
    plt.title('Выборка s1')

    plt.subplot(2, 2, 2)
    plt.hist(s2, edgecolor='black', bins=8)
    plt.title('Выборка s2')

    plt.show()

    r1 = pearsonr(s, s1)[0]
    r2 = pearsonr(s, s2)[0]
    r3 = pearsonr(s1, s2)[0]

    # проверка значимости
    validate(r1, n)
    validate(r2, n)
    validate(r3, n)
    print(".......................")

    # подсчет коэффициентов корреляций через ковариации
    get_r_from_cov(s, s1)
    get_r_from_cov(s, s2)
    get_r_from_cov(s1, s2)
    print(".......................")

    # корреляционная матрица
    total = np.array([s, s1, s2])
    cor_mat = np.corrcoef(total)
    print(cor_mat)

np.set_printoptions(precision=3)
#запустить часть 1 - гистограммы и выборочные характеристики
#descr_stat_test()
correlation_test()