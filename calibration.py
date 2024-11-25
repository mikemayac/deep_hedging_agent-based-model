import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from helpers import *
import pickle

if __name__ == "__main__":
    import sys

    IDX = int(sys.argv[1])
    TOTAL = int(sys.argv[2])
    print("index is %d; total is %d" % (IDX, TOTAL))

    kappa_list = np.arange(0.01, 0.15, 0.01)
    beta_lf_list = np.arange(0.01, 0.1, 0.01)
    # beta_hf_list = np.arange(0.01, 0.2, 0.01)
    beta_hf_list = [0.]
    sigma_N_list = np.arange(2.3, 3., 0.05)
    # sigma_V_list = np.arange(0.7, 1.3, 0.1)
    # window_list = np.arange(30, 50, 5)
    # window = 40
    lam_list = np.arange(0.02, 0.035, 0.002)
    # For extended Chiarella model
    # kappa_list = np.arange(0.3, 0.8, 0.1)
    # beta_lf_list = np.arange(0.01, 0.03, 0.01)
    # # beta_hf_list = np.arange(0.01, 0.2, 0.01)
    # beta_hf_list = [0.]
    # sigma_N_list = np.arange(0.1, 0.65, 0.05)
    # # sigma_V_list = np.arange(0.7, 1.3, 0.1)
    # # window_list = np.arange(30, 50, 5)
    # # window = 40
    # lam_list = np.arange(0.005, 0.08, 0.005)

    value_list = []
    for kappa in kappa_list:
        for beta_lf in beta_lf_list:
            for beta_hf in beta_hf_list:
                for sigma_N in sigma_N_list:
                    # for sigma_V in sigma_V_list:
                    # for window in window_list:
                    for lam in lam_list:
                        value_list.append([kappa, beta_lf, beta_hf, sigma_N, lam])
                        # value_list.append([kappa, beta_lf, beta_hf, sigma_N, sigma_V, lam])

    print(len(value_list))
    length = len(value_list) // TOTAL + 1
    left, right = IDX * length, (IDX + 1) * length
    to_eval = value_list[left:right]
    print("number to evaluate: %d" % len(to_eval))

    # data = yf.download('^GSPC','2011-02-01','2022-12-31') # Previous
    # data = yf.download('^GSPC','1999-08-26','2011-07-29')
    data = yf.download('^N225', '1999-07-08', '2011-09-29')
    print(len(data))
    close = data['Adj Close'].values
    his_ret = close[1:] / close[:-1] - 1
    his_ret2 = his_ret ** 2
    print(len(his_ret))

    # LAGS = [1,2,3,10,11,12,20,21,22]
    LAGS = [1, 2, 3]
    LAGS2 = [i for i in np.arange(1, 10)]
    num_sim = 10
    np_seed = 1
    np.random.seed(np_seed)
    num_period = 3000
    # g = np.mean(his_ret)
    # g = 2.9e-5 # (r - 0.5* (np.std(his_ret) * np.sqrt(250))**2) / 250
    # sigma_V = 0.0112 # np.std(his_ret) * np.sqrt(250) * np.sqrt(1./250)
    r = 0.03
    g = (r - 0.5 * (np.std(his_ret) * np.sqrt(250)) ** 2) / 250
    sigma_V = np.std(his_ret) * np.sqrt(250) * np.sqrt(1. / 250)
    gamma = calc_gamma(his_ret)

    res_record = {}
    counter = 0
    for item in to_eval:
        # kappa, beta_lf, beta_hf, sigma_N, sigma_V, lam = item
        kappa, beta_lf, beta_hf, sigma_N, lam = item
        params = {'kappa': kappa, 'beta_lf': beta_lf, 'beta_hf': beta_hf, 'sigma_N': sigma_N, 'sigma_V': sigma_V,
                  'g': g, 'v0': np.log(100.)}

        sim_params = {'n_runs': num_sim, 'seed': np_seed, 'dt': 1, 'T': num_period, 'lambda': lam, 'p0': np.log(100.),
                      'alpha_hf': .99, 'alpha_lf': 1. / (1. + 5.), 'gamma': gamma}

        dist = func_opt(params, sim_params, his_ret, LAGS, LAGS2)
        tmp = {'kappa': kappa, 'beta_lf': beta_lf, 'beta_hf': beta_hf, 'sigma_N': sigma_N, 'lambda': lam, 'dist': dist}
        res_record[left + counter] = tmp
        counter += 1
        if counter % 50 == 0:
            print("index: %d, counter: %d" % (IDX, counter))
    file_name = "./records/records_%d.pkl" % IDX
    with open(file_name, 'wb') as f:
        pickle.dump(res_record, f)
        f.close()





