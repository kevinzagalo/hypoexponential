from numpy import prod, zeros, random, array, cumsum, sqrt
from math import exp


class Hypoexponential:
    # only works if the parameters eta are distinct
    def __init__(self, eta):
        self._eta = eta
        self._prod_eta = []
        self._weights = []
        for i, eta_i in enumerate(self._eta):
            tmp_list = list(self._eta[:i])
            tmp_list.extend(self._eta[i+1:])
            self._prod_eta.append(prod([(eta_j - eta_i) for eta_j in tmp_list]))
            self._weights.append(prod([1 / (1 - eta_i/eta_j) for eta_j in tmp_list]))

    def pdf(self, x):
        return [prod(self._eta) * sum([exp(-eta_i * xx) / self._prod_eta[i]
                                      for i, eta_i in enumerate(self._eta)]) for xx in x]

    def cdf(self, x):
        return [sum([(1-exp(-eta_j * xx)) * self._weights[j] for j, eta_j in enumerate(self._eta)]) for xx in x]

    @property
    def weights(self):
        return self._weights

    @property
    def params(self):
        return {'rates': self._eta}

    def sample(self, n_sample=1):
        S = zeros((n_sample, len(self._eta)))
        for i, eta_i in enumerate(self._eta):
            S[:, i] = random.exponential(1/eta_i, size=n_sample)
        return S.sum(axis=1)


if __name__ == '__main__':
    from numpy import linspace, array
    import matplotlib.pyplot as plt

    eta = [0.05665075, 0.07182203, 0.05739665]
           #0.06123506, 0.0448171, 0.06252558,
           #0.05692616, 0.04154262, 0.07924733,
           #0.03371705, 0.03181677, 0.05002229,
           #0.06976966, 0.06164375]#, 0.06344596]
           #0.03570836, 0.063677, 0.04330521,
           #0.03788124, 0.04233851, 0.10019797,
           #0.0703162, 0.0443816, 0.05545152,
           #0.04114605, 0.02203691, 0.02705287,
           #0.0388603, 0.04493987]
    eta = list(set(eta))

    sum_exp = Hypoexponential(eta)
    sample = sum_exp.sample(1000)
    t_range = linspace(0.001, max(sample))
    plt.hist(sample, bins=50, density=True)
    plt.plot(t_range, sum_exp.pdf(t_range))
    plt.show()

    print(sum_exp.params)
    print(sum(sum_exp.weights))
    plt.plot(t_range, sum_exp.cdf(t_range))
    plt.show()
