import pickle
import os
from collections import defaultdict
from multiprocessing import Pool, TimeoutError
import logging

import pystan
import numpy as np
from dask import delayed, compute
from scipy.stats import chi2
from statsmodels.stats import multitest

from utils import *


# class/method to suppress pystan outputs
class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def nb_add_q_values(diff_intron_dict, error_rate, method):
    p_values = []
    indices = {}
    index = 0
    q_values = []
    for i, (p_value, _, _) in enumerate(diff_intron_dict.values()):
        q_values.append(None)
        if p_value is None:
            continue
        p_values.append(p_value)
        indices[index] = i
        index += 1

    fdr_results = multitest.multipletests(p_values, alpha=error_rate, method=method)
    _q_values = fdr_results[1].tolist()

    for j, i in indices.items():
        q_values[i] = _q_values[j]

    for i, coord in enumerate(diff_intron_dict.keys()):
        diff_intron_dict[coord].append(q_values[i])

###############################################################################
## Zero inflated Negative Binomial model
def NB_model(df, conditions, model_dir, num_workers=4, count=5, error_rate=0.05, method='fdr_bh', batch_size=500):
    diff_intron_dict = {}
    pred_intron_dict = {}
    coords_batches = []
    delayed_results = []

    ys = []
    coords = []
    # ys_list = []
    for i, row in enumerate(df.itertuples()):
        row_list = list(row)
        coord = row_list[0]
        ys.append(np.array(row_list[1:-1], dtype=np.int))
        coords.append(coord)
        if i > 0 and i % batch_size == 0:
            delayed_results.append(delayed(batch_run_NB_model)(ys, conditions, model_dir, count))
            # ys_list.append(ys)
            coords_batches.append(coords)
            coords = []
            ys = []

    if len(ys) > 0:
        delayed_results.append(delayed(batch_run_NB_model)(ys, conditions, model_dir, count))
        # ys_list.append(ys)
        coords_batches.append(coords)

    results_batches = list(compute(*delayed_results, traverse=False, num_workers=num_workers, scheduler="processes"))
    # results_batches = multiprocesses_run_NB_model(ys_list, conditions, count, model_dir, num_workers)

    sig_coords = []
    for coords, results in zip(coords_batches, results_batches):
        for coord, result in zip(coords, results):
            p_value, log_likelihood, mus = result
            diff_intron_dict[coord] = [p_value, log_likelihood, mus]
            pred_intron_dict[coord] = 'LO_DATA'
            if p_value is not None and np.any(mus >= 1):
                sig_coords.append(coord)
                pred_intron_dict[coord] = 'OK'

    df.loc[sig_coords, 'label'] = 1

    logging.info(f"{i+1} introns processed")
    nb_add_q_values(diff_intron_dict, error_rate, method)

    return diff_intron_dict, pred_intron_dict


def multiprocesses_run_NB_model(ys_list, conditions, count, model_dir, num_workers):
    results = []
    pool = Pool(processes=num_workers)
    workers = []
    for ys in ys_list:
        workers.append(pool.apply_async(batch_run_NB_model, (ys, conditions, model_dir, count)))

    for worker in workers:
        results.append(worker.get())
    pool.close()
    pool.terminate()
    pool.join()
    return results


def batch_run_NB_model(ys, conditions, model_dir, count):
    null_model = pickle.load(open(model_dir / 'null_NB_model.pkl', 'rb'))
    alt_model = pickle.load(open(model_dir / 'alt_NB_model.pkl', 'rb'))
    results = []
    for y in ys:
        results.append(run_NB_model(y, conditions, count, null_model, alt_model))
    return results


def run_NB_model(y, conditions, count, null_model, alt_model):
    N = y.shape[0]
    z = conditions
    K = z.shape[1]
    # init null model
    mu_raw = np.mean(y)
    null_data_dict = {'N': N, 'y': y, 'mu_raw': np.median(y)}
    # init alternative model
    mu_raw = []
    means = []
    _vars = []
    for k in range(K):
        indices = np.where(z[:, k] > 0)[0]
        mu_raw.append(np.median(np.take(y, indices)))
        means.append(np.mean(np.take(y, indices)))
        _vars.append(np.var(np.take(y, indices)))

    values = np.array(means) - np.array(_vars)
    if max(means) <= count and np.any(values < 0):
        return None, None, np.array(means)

    alt_data_dict = {'N': N, 'K': K, 'y': y, 'z': z, 'mu_raw': mu_raw}
    with suppress_stdout_stderr():
        i = 0
        while i < 10:
            try:
                fit_null = null_model.optimizing(data=null_data_dict, as_vector=False, init_alpha=1e-5)
                fit_alt = alt_model.optimizing(data=alt_data_dict, as_vector=False, init_alpha=1e-5)
            except RuntimeError:
                i += 1
                continue
            break

    if i == 10:
        return None, None, np.array(means)
    else:
        log_likelihood = fit_alt['value'] - fit_null['value']
        mus = fit_alt['par']['mu']
        p_value = 1 - chi2(3 * (K - 1)).cdf(2 * log_likelihood)
    return p_value, log_likelihood, mus


def init_null_BN_model(model_dir):
    code = """
    data {
        int<lower=1> N;
        int<lower=0> y[N];
        real<lower=0> mu_raw;
    }
    parameters {
        real<lower=0, upper=1> theta;
        real<lower=0> mu;
        real<lower=1e-4> inverted_phi;
    }
    model {
        mu ~ normal(mu_raw, sqrt(mu_raw/10)+1e-4);
        inverted_phi ~ cauchy(1e-2, 5);
        for (n in 1:N) {
            if (y[n] == 0){
                target += log_sum_exp(bernoulli_lpmf(1 | theta),
                                         bernoulli_lpmf(0 | theta)
                                         + neg_binomial_2_lpmf(y[n] | mu+1e-4, 1./inverted_phi));
            }
            else{
                target += bernoulli_lpmf(0 | theta) + neg_binomial_2_lpmf(y[n] | mu+1e-4, 1./inverted_phi);
            }
        }
    }
    """
    file = model_dir / 'null_NB_model.pkl'
    if not file.exists():
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
        model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
        with open(file, 'wb') as f:
            pickle.dump(model, f)


def init_alt_BN_model(model_dir):
    code = """
    data {
        int<lower=1> N;
        int<lower=1> K;
        int<lower=0> y[N];
        int<lower=0> z[N, K];
        vector<lower=0>[K] mu_raw;
    }
    parameters {
        real<lower=0, upper=1> theta[K];
        vector<lower=0>[K] mu;
        vector<lower=1e-4>[K] inverted_phi;
    }
    model {
        mu ~ normal(mu_raw, sqrt(mu_raw/10)+1e-4);
        inverted_phi ~ cauchy(1e-2, 5);
        for (n in 1:N) {
            vector[K] lps;
            if (y[n] == 0){
                for (k in 1:K){
                    lps[k] = log_sum_exp(bernoulli_lpmf(1 | theta[k]),
                                         bernoulli_lpmf(0 | theta[k])
                                         + neg_binomial_2_lpmf(y[n] | mu[k]+1e-4, 1./inverted_phi[k]));
                    lps[k] *= z[n][k];
                }
            }
            else{
                for (k in 1:K) {
                    lps[k] = bernoulli_lpmf(0 | theta[k]) + neg_binomial_2_lpmf(y[n] | mu[k]+1e-4, 1./inverted_phi[k]);
                    lps[k] *= z[n][k];
                }
            }
            target += lps;
        }
    }
    """
    file = model_dir / 'alt_NB_model.pkl'
    if not file.exists():
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
        model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
        with open(file, 'wb') as f:
            pickle.dump(model, f)


###############################################################################
def dm_add_q_values(diff_dm_group_dict, error_rate, method):
    p_values = [v[1] for v in diff_dm_group_dict.values()]
    fdr_results = multitest.multipletests(p_values, alpha=error_rate, method=method)
    q_values = fdr_results[1].tolist()

    for i, coord in enumerate(diff_dm_group_dict.keys()):
        diff_dm_group_dict[coord].append(q_values[i])

    return diff_dm_group_dict


## Dirichlet Multinomial model
def DM_model(df, index_df, conditions, model_dir, num_workers=4,
            error_rate=0.05, method='fdr_bh', batch_size=1000):
    _df = df[df['label'] == 1].drop(['label'], axis=1)
    diff_dm_intron_dict = defaultdict(list)
    groups = []
    coords_list = []
    delayed_results = []

    indices = _df.index.tolist()
    _index_df = index_df.loc[index_df['index'].isin(indices)]
    chr_strand_pred_diff_introns_dict = defaultdict(lambda: defaultdict(set))
    chrs = list(_index_df.index.unique(level='chromosome'))

    i = 0
    coords_batches = []
    groups_batches = []
    ys = []
    groups = []
    coords = []
    # ys_list = []
    for _chr in chrs:
        strands = list(_index_df.loc[_chr].index.unique(level='strand'))
        for strand in strands:
            group_dict = get_splice_site_groups(_index_df.loc[_chr].loc[strand]['index'].tolist())
            for group, intron_coords in group_dict.items():
                if len(intron_coords) > 1:
                    ys.append(_df.loc[intron_coords].values.T.astype(int))
                    groups.append((f"g{i+1:06d}", (_chr, strand, group[0], group[1])))
                    coords.append(intron_coords)
                    if i > 0 and i % batch_size == 0:
                        delayed_results.append(delayed(batch_run_DM_model)(ys, conditions, model_dir))
                        # ys_list.append(ys)
                        groups_batches.append(groups)
                        coords_batches.append(coords)
                        groups = []
                        coords = []
                        ys = []
                    i += 1

    if len(ys) > 0:
        delayed_results.append(delayed(batch_run_DM_model)(ys, conditions, model_dir))
        # ys_list.append(ys)
        groups_batches.append(groups)
        coords_batches.append(coords)

    results_batches = list(compute(*delayed_results, traverse=False, num_workers=num_workers, scheduler="processes"))
    # results_batches = multiprocesses_run_DM_model(ys_list, conditions, model_dir, num_workers)

    diff_dm_group_dict = {}
    for groups, results, coords_list in zip(groups_batches, results_batches, coords_batches):
        for group_info, result, coords in zip(groups, results, coords_list):
            group_id, group = group_info
            p_value, log_likelihood, psis = result
            diff_dm_group_dict[group] = [group_id, p_value, log_likelihood, coords]
            for coord, psi in zip(coords, psis):
                diff_dm_intron_dict[coord].append((group_id, psi))

    logging.info(f"{i+1} groups processed")
    diff_dm_group_dict = dm_add_q_values(diff_dm_group_dict, error_rate, method)
    return diff_dm_intron_dict, diff_dm_group_dict


def multiprocesses_run_DM_model(ys_list, conditions, model_dir, num_workers):
    results = []
    pool = Pool(processes=num_workers)
    workers = []
    for ys in ys_list:
        workers.append(pool.apply_async(batch_run_DM_model, (ys, conditions, model_dir)))

    for worker in workers:
        results.append(worker.get())
    pool.close()
    pool.terminate()
    pool.join()
    return results


def batch_run_DM_model(ys, conditions, model_dir):
    null_model = pickle.load(open(model_dir / 'null_DM_model.pkl', 'rb'))
    alt_model = pickle.load(open(model_dir / 'alt_DM_model.pkl', 'rb'))
    results = []
    for y in ys:
        results.append(run_DM_model(y, conditions, null_model, alt_model))

    return results


def run_DM_model(y, conditions, null_model, alt_model):
    N, M = y.shape
    z = conditions
    K = z.shape[1]

    # init null model
    null_data_dict = {'N': N, 'M': M, 'y': y}

    # init alternative model
    alt_data_dict = {'N': N, 'M': M, 'K': K, 'y': y, 'z': z}

    with suppress_stdout_stderr():
        i = 0
        while i < 10:
            try:
                fit_null = null_model.optimizing(data=null_data_dict, as_vector=False, init_alpha=1e-5)
                fit_alt = alt_model.optimizing(data=alt_data_dict, as_vector=False, init_alpha=1e-5)
            except RuntimeError:
                i += 1
                continue
            break

    if i == 10:
        return None, None, None
    else:
        log_likelihood = fit_alt['value'] - fit_null['value']
        psis = fit_alt['par']['alpha'].T.tolist()
        p_value = 1 - chi2(M * (K - 1)).cdf(2 * (log_likelihood))
        return p_value, log_likelihood, psis


def init_null_DM_model(model_dir):
    code = """
    functions {
        real dirichlet_multinomial_lpmf(int[] y, vector alpha) {
            real alpha_plus = sum(alpha);
            return lgamma(alpha_plus) + sum(lgamma(alpha + to_vector(y)))
                        - lgamma(alpha_plus + sum(y)) - sum(lgamma(alpha));
        }
    }
    data {
        int<lower=1> N;
        int<lower=1> M;
        int<lower=0> y[N, M];
    }
    parameters {
        simplex[M] alpha;
        real<lower=0> conc;
    }
    model {
        conc ~ normal(0, 1e6);
        for (n in 1:N) {
            target += dirichlet_multinomial_lpmf(y[n] | conc * alpha);
        }
    }
    """
    file = model_dir / 'null_DM_model.pkl'
    if not file.exists():
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
        model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
        with open(file, 'wb') as f:
            pickle.dump(model, f)


def init_alt_DM_model(model_dir):
    code = """
    functions {
        real dirichlet_multinomial_lpmf(int[] y, vector alpha) {
            real alpha_plus = sum(alpha);
            return lgamma(alpha_plus) + sum(lgamma(alpha + to_vector(y)))
                        - lgamma(alpha_plus + sum(y)) - sum(lgamma(alpha));
        }
    }
    data {
        int<lower=1> N;
        int<lower=1> M;
        int<lower=1> K;
        int<lower=0> y[N, M];
        int<lower=0> z[N, K];
    }
    parameters {
        simplex[M] alpha[K];
        real<lower=0> conc;
    }
    model {
        conc ~ normal(0, 1e6);
        for (n in 1:N) {
            vector[K] lps;
            for (k in 1:K){
                lps[k] = dirichlet_multinomial_lpmf(y[n] | conc * alpha[k]);
                lps[k] *= z[n][k];
            }
            target += lps;
        }
    }
    """
    file = model_dir / 'alt_DM_model.pkl'
    if not file.exists():
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
        model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
        with open(file, 'wb') as f:
            pickle.dump(model, f)

