import numpy as np
import pandas as pd
import scipy

VOL = 0.2515
RHO = -0.382
VOLVOL = 0.07


def hill_estimator(data, tail=0.05):
    """
    Returns the Hill Estimators for some 1D data set.
    """
    data = abs(data)
    Y = np.sort(data)[::-1]
    k = int(tail * len(Y))
    while Y[k] < 1e-20:
        tail -= 0.01
        if tail <= 0.02:
            return 1.
        k = int(tail * len(Y))
    tmp = Y[:k]
    summ = np.sum(np.log(tmp / tmp[-1]))
    hill_est = (1. / k) * summ
    return hill_est


def get_auto_correl(X, lags):
    c = {}
    for lag in lags:
        c[lag] = np.corrcoef(X[:-lag], X[lag:])[0][1]

    return pd.Series(c)


# Distance Calculation
# Volatility
def volatility_diff(his_ret, sim_ret):
    scale = np.sqrt(252)
    diff = np.std(sim_ret) * scale - np.std(his_ret) * scale
    return abs(diff)


# Auto-correlation diff
# For return first order acf, use small lags
def acf_diff(his_ret, sim_ret, lags):
    acf1 = get_auto_correl(his_ret, lags)
    acf2 = get_auto_correl(sim_ret, lags)
    diff = np.abs(acf1 - acf2)
    return np.mean(diff)


# TODO: weights according to bootstrapped variance
def prepare_distance(his_ret, sim_ret, lags, lags2):
    vol_diff = volatility_diff(his_ret, sim_ret)
    # cdf_ks = ks_2samp(his_ret, sim_ret).statistic
    fat_tail = abs(hill_estimator(his_ret) - hill_estimator(sim_ret))
    ret1_acf_diff = acf_diff(his_ret, sim_ret, lags=lags)
    ret2_acf_diff = acf_diff(his_ret ** 2, sim_ret ** 2, lags=lags2)

    return np.array([vol_diff, fat_tail, ret1_acf_diff, ret2_acf_diff])
    # return np.array([vol_diff, ret2_acf_diff])
    # return np.array([ret2_acf_diff])


def run_mt1(sim_params, model_params, v=None, fix_gamma=True):
    n_runs = sim_params['n_runs']
    seed = sim_params['seed']
    dt = sim_params['dt']
    T = sim_params['T']

    np.random.seed(sim_params['seed'])

    n_steps = int(T / dt)
    p = np.zeros((n_steps, n_runs)) * np.nan
    d_ft = np.zeros((n_steps, n_runs)) * np.nan
    d_mt_st = np.zeros((n_steps, n_runs)) * np.nan
    d_mt_lt = np.zeros((n_steps, n_runs)) * np.nan
    d_nt = np.zeros((n_steps, n_runs)) * np.nan
    p[0] = np.ones(n_runs) * sim_params['p0']

    mlt = np.zeros((n_steps, n_runs)) * np.nan
    mlt[0] = np.ones(n_runs) * np.nan

    mst = np.zeros((n_steps, n_runs)) * np.nan
    mst[0] = np.ones(n_runs) * np.nan

    if v is None:
        e_v = np.random.normal(model_params['g'], model_params['sigma_V'] * np.sqrt(dt),
                               size=(n_steps, n_runs))
        e_v[0] = model_params['v0'] if 'v0' in model_params else sim_params['p0']
        v = np.cumsum(e_v, axis=0)

    gamma = sim_params['gamma'] if fix_gamma else model_params['gamma']
    for t in np.arange(1, n_steps):
        _d_ft = model_params['kappa'] * (v[t] - p[t - 1]) * dt
        _d_mt_lt = (model_params['beta_lf'] * np.tanh(gamma * mlt[t - 1]) * dt if t > 1 else 0)
        _d_mt_st = (model_params['beta_hf'] * np.tanh(gamma * mst[t - 1]) * dt if t > 1 else 0)
        _d_nt = np.random.normal(0, 1, size=n_runs) * model_params['sigma_N'] * np.sqrt(dt)
        d_ft[t] = _d_ft
        d_mt_lt[t] = _d_mt_lt if t > 1 else np.nan
        d_mt_st[t] = _d_mt_st if t > 1 else np.nan
        d_nt[t] = _d_nt
        dD = _d_ft + _d_mt_st + _d_mt_lt + _d_nt
        dp = dD * sim_params['lambda']
        p[t] = p[t - 1] + dp
        mst[t] = (1. - sim_params['alpha_hf']) * mst[t - 1] + sim_params['alpha_hf'] * dp if t > 1 else dp
        mlt[t] = (1. - sim_params['alpha_lf']) * mlt[t - 1] + sim_params['alpha_lf'] * dp if t > 1 else dp

    ts = np.linspace(0, T - dt, n_steps)

    return {'M_HF': pd.DataFrame(mst, index=ts), 'M_LF': pd.DataFrame(mlt, index=ts),
            'P': pd.DataFrame(p, index=ts), 'V': pd.DataFrame(v, index=ts),
            'D_MT_HF': pd.DataFrame(d_mt_st, index=ts), 'D_MT_LF': pd.DataFrame(d_mt_lt, index=ts),
            'D_FT': pd.DataFrame(d_ft, index=ts), 'D_NT': pd.DataFrame(d_nt, index=ts)}


def run_mt2(sim_params, model_params, v=None, fix_gamma=True):
    n_runs = sim_params['n_runs']
    seed = sim_params['seed']
    dt = sim_params['dt']
    T = sim_params['T']

    np.random.seed(sim_params['seed'])

    n_steps = int(T / dt)
    p = np.zeros((n_steps, n_runs)) * np.nan
    d_ft = np.zeros((n_steps, n_runs)) * np.nan
    d_mt_st = np.zeros((n_steps, n_runs)) * np.nan
    d_mt_lt = np.zeros((n_steps, n_runs)) * np.nan
    d_nt = np.zeros((n_steps, n_runs)) * np.nan
    p[0] = np.ones(n_runs) * sim_params['p0']

    mlt = np.zeros((n_steps, n_runs)) * np.nan
    mlt[0] = np.ones(n_runs) * np.nan

    mst = np.zeros((n_steps, n_runs)) * np.nan
    mst[0] = np.ones(n_runs) * np.nan

    if v is None:
        e_v = np.random.normal(model_params['g'], model_params['sigma_V'] * np.sqrt(dt),
                               size=(n_steps, n_runs))
        e_v[0] = model_params['v0'] if 'v0' in model_params else sim_params['p0']
        v = np.cumsum(e_v, axis=0)

    # window = model_params['window']
    # noise = np.random.normal(0, 1, size=(n_steps + window - 1, n_runs))
    # noise = np.cumsum(noise, axis=0)
    # noise = scipy.signal.convolve2d(noise, np.ones((window, 1)), mode='valid')
    # print(v.shape, noise.shape)

    kappa_noise = 0.005
    sigma_noise = VOLVOL
    theta_noise = VOL ** 2  # history std ** 2
    rho = RHO

    noise = np.zeros((n_steps, n_runs)) * np.nan
    noise[0, :] = theta_noise

    gamma = sim_params['gamma'] if fix_gamma else model_params['gamma']
    for t in np.arange(1, n_steps):
        _d_ft = model_params['kappa'] * (v[t] - p[t - 1]) * dt
        _d_mt_lt = (model_params['beta_lf'] * np.tanh(gamma * mlt[t - 1]) * dt if t > 1 else 0)
        _d_mt_st = (model_params['beta_hf'] * np.tanh(gamma * mst[t - 1]) * dt if t > 1 else 0)

        z1 = np.random.randn(n_runs)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.randn(n_runs)
        noise[t, :] = noise[t - 1, :] + kappa_noise * (theta_noise - noise[t - 1, :]) * dt + sigma_noise * np.sqrt(
            np.abs(noise[t - 1, :]) * dt) * z1
        # _d_nt = z2 * model_params['sigma_N'] * np.sqrt(dt) * np.sqrt(noise[t])
        _d_nt = z2 * model_params['sigma_N'] * np.sqrt(dt) * np.sqrt(np.abs(noise[t]))
        d_ft[t] = _d_ft
        d_mt_lt[t] = _d_mt_lt if t > 1 else np.nan
        d_mt_st[t] = _d_mt_st if t > 1 else np.nan
        d_nt[t] = _d_nt
        dD = _d_ft + _d_mt_st + _d_mt_lt + _d_nt
        dp = dD * sim_params['lambda']
        p[t] = p[t - 1] + dp
        mst[t] = (1. - sim_params['alpha_hf']) * mst[t - 1] + sim_params['alpha_hf'] * dp if t > 1 else dp
        mlt[t] = (1. - sim_params['alpha_lf']) * mlt[t - 1] + sim_params['alpha_lf'] * dp if t > 1 else dp

    ts = np.linspace(0, T - dt, n_steps)

    return {'M_HF': pd.DataFrame(mst, index=ts), 'M_LF': pd.DataFrame(mlt, index=ts),
            'P': pd.DataFrame(p, index=ts), 'V': pd.DataFrame(v, index=ts),
            'D_MT_HF': pd.DataFrame(d_mt_st, index=ts), 'D_MT_LF': pd.DataFrame(d_mt_lt, index=ts),
            'D_FT': pd.DataFrame(d_ft, index=ts), 'D_NT': pd.DataFrame(d_nt, index=ts)}


def func_opt(params, sim_params, his_ret, lags, lags2):
    # res = run_mt1(sim_params, params, v=None)
    res = run_mt2(sim_params, params, v=None)
    # price_arr = res['P'].values
    price_arr = np.exp(res['P'].values)
    dist_list = []
    for idx in range(price_arr.shape[1]):
        price = price_arr[:, idx]
        sim_ret = price[1:] / price[:-1] - 1
        tmp = np.sum(prepare_distance(his_ret, sim_ret, lags, lags2))
        dist_list.append(tmp)
    # print(np.round(dist_list, 4))
    return np.mean(dist_list)


def calc_gamma(his_ret, alpha=1 / 6):
    m_arr = [0 for _ in range(len(his_ret))]
    m_arr[0] = his_ret[0]
    for i in range(1, len(his_ret)):
        m_arr[i] = (1 - alpha) * m_arr[i - 1] + alpha * his_ret[i]
    gamma = 1. / (2 * np.std(m_arr))
    return gamma