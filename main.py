import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm
from scipy import optimize

def manager_problem(x, y, t, params, a, C, sigma):
    # Unpacking parameters
    Ac, Ad, lc, ld, sc, sk, sd, beta, r = params
    a1, a2 = a
    sigmac, sigmad = sigma
    # Unpacking t
    tc, td, tk = t
    # Net production
    x1 = (1 + Ac * np.exp(-lc * x ** 2)) * (sc * tc + sk * tk) + x
    x2 = (1 + Ad * np.exp(-ld * y ** 2)) * (sd * td + sk * tk) + y
    w = a1 * x1 + a2 * x2 + beta

    return x1, x2, w

def pre_integration(x, y, t, params, a, C, sigma):
    r = params[-1]
    sigmac, sigmad = sigma
    w = manager_problem(x, y, t, params, a, C, sigma)[-1]
    return np.exp(-r * (w - t.T @ C @ t)) \
            * norm.pdf(x, loc = 0, scale = sigmac) \
            * norm.pdf(y, loc = 0, scale = sigmad)

def payoff(t, params, a, C, sigma):
    r = params[-1]
    sigmac, sigmad = sigma
    integral, _ = dblquad(lambda x, y: pre_integration(x, y, t, params, a, C, sigma),
                       -3 * sigmac, 3 * sigmac, -3 * sigmad, 3 * sigmad)
    return - np.log(integral) / r

def optimal_t(params, a, C, sigma):
    return optimize.minimize(lambda t: -payoff(t, params, a, C, sigma), [1, 1, 1],
                             method = "SLSQP")#,
                             #bounds = optimize.Bounds([0, 0, 0], [np.inf, np.inf, np.inf]))

def principal_problem(b, params, C, sigma):
    def optimization_fun(x, y, a, b, params, C, sigma):
        t = optimal_t(params, a, C, sigma).x
        sigmac, sigmad = sigma
        a1, a2 = a
        r = params[-1]
        b1, b2 = b
        man_q = manager_problem(x, y, t, params, a, C, sigma)
        B = b1 * man_q[0] + b2 * man_q[1] \
            * norm.pdf(x, loc = 0, scale = sigmac) \
            * norm.pdf(y, loc = 0, scale = sigmad)
        return B - t.T @ C @ t - r * (a1 ** 2 * sigmac**2 + a2 ** 2 + sigmad ** 2)

    def integration(a,b, params, C, sigma):
        sigmac, sigmad = sigma
        integral, _ = dblquad(lambda x, y: optimization_fun(x, y, a, b, params, C, sigma),
                       -3 * sigmac, 3 * sigmac, -3 * sigmad, 3 * sigmad)
        return integral

    return optimize.minimize(lambda a: -integration(a, b, params, C, sigma), [1, 1],
                             method="SLSQP", options={"maxiter":1})

if __name__ == "__main__":
    params = [1, 1, 1, 1, 1, 1, 2, 1, 1]
    sigma = [1, 1]
    a = [1, 1]
    b = [1, 1]
    t = np.array([1, 1, 1])
    C = np.array([[2, 0, -1], [0, 2, -1], [-1, -1, 2]])
    #print(optimal_t(params, a, C, sigma))
    print(principal_problem(b, params, C, sigma))
