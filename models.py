import pickle
import os
from collections import defaultdict
import logging
from pathlib import Path

import pystan
import numpy as np
from dask import delayed, compute
from scipy.stats import chi2
from statsmodels.stats import multitest
import pandas


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
    for i, (p_value, _, _, _) in enumerate(diff_intron_dict.values()):
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
def NB_model(df, conditions, confounders, model_dir, num_workers=4, count=5, error_rate=0.05,
             method='fdr_bh', batch_size=500, aggressive_mode=False):
    diff_intron_dict = {}
    pred_intron_dict = {}
    coords_batches = []
    delayed_results = []

    ys = []
    coords = []
    for i, row in enumerate(df.itertuples()):
        row_list = list(row)
        coord = row_list[0]
        ys.append(np.array(row_list[1:-1], dtype=np.int))
        coords.append(coord)
        if i > 0 and i % batch_size == 0:
            delayed_results.append(delayed(batch_run_NB_model)(ys, conditions, confounders, model_dir, count, aggressive_mode))
            coords_batches.append(coords)
            coords = []
            ys = []

    if len(ys) > 0:
        delayed_results.append(delayed(batch_run_NB_model)(ys, conditions, confounders, model_dir, count, aggressive_mode))
        coords_batches.append(coords)

    results_batches = list(compute(*delayed_results, traverse=False, num_workers=num_workers, scheduler="processes"))

    sig_coords = []
    for coords, results in zip(coords_batches, results_batches):
        for coord, result in zip(coords, results):
            p_value, log_likelihood, mus, sigmas = result
            diff_intron_dict[coord] = [p_value, log_likelihood, mus, sigmas]
            pred_intron_dict[coord] = 'LO_DATA'
            if p_value is not None and np.any(mus >= 1):
                sig_coords.append(coord)
                pred_intron_dict[coord] = 'OK'

    df.loc[sig_coords, 'label'] = 1

    logging.info(f"{i+1} introns processed")
    nb_add_q_values(diff_intron_dict, error_rate, method)

    return diff_intron_dict, pred_intron_dict

def batch_run_NB_model(ys, conditions, confounders, model_dir, count, aggressive_mode):
    null_model = pickle.load(open(Path(model_dir) / 'null_NB_model.cov.pkl', 'rb'))
    alt_model = pickle.load(open(Path(model_dir) / 'alt_NB_model.cov.pkl', 'rb'))
    results = []
    for y in ys:
        results.append(run_NB_model(y, conditions, confounders, count, null_model, alt_model, aggressive_mode))
    return results


def custom_mean(y, aggressive_mode):
    if aggressive_mode:
        return np.mean(y)
    else:
        return np.median(y)


def custom_count(quantiles, means, aggressive_mode):
    if aggressive_mode:
        return max(quantiles)
    else:
        return max(means)


def run_NB_model(y, conditions, confounders, count, null_model, alt_model, aggressive_mode):

    N = y.shape[0]
    z = conditions
    K = z.shape[1]
    x_null=confounders.drop(['condition'],axis=1)
    x_alt=confounders
    # init null model
    null_data_dict = {'N': N, 'y': y, 'mu_raw': custom_mean(y, aggressive_mode), 'K': K, 'P': len(x_null.columns), 'x': x_null}

    # init alternative model
    mu_raw = []
    means = []
    _vars = []
    quantiles = []
    for k in range(K):
        indices = np.where(z[:, k] > 0)[0]
        array = np.take(y, indices)
        mu_raw.append(custom_mean(array, aggressive_mode))
        means.append(np.mean(array))
        num = 0.9 if indices.shape[0] >= 30 else 0.
        quantiles.append(np.quantile(array, num))
        _vars.append(np.var(array))

    _count = custom_count(quantiles, means, aggressive_mode)
    values = np.array(means) - np.array(_vars)
    if _count <= count and np.any(values < 0):
        return None, None, np.array(means), np.array(_vars)

    alt_data_dict = {'N': N, 'K': K, 'y': y, 'z': z, 'mu_raw': mu_raw, 'P': len(x_alt.columns), 'x': x_alt}

    max_optim_n=10
    with suppress_stdout_stderr():
        i = 0
        while i < max_optim_n:
            try:
                fit_null = null_model.optimizing(data=null_data_dict, as_vector=False, init_alpha=1e-5)
                fit_alt = alt_model.optimizing(data=alt_data_dict, as_vector=False, init_alpha=1e-5)
            except RuntimeError:
                i += 1
                continue
            break

    if i == max_optim_n:
        return None, None, np.array(means), None
    else:
        log_likelihood = fit_alt['value'] - fit_null['value']
        mus = fit_alt['par']['mu']
        reciprocal_phi = fit_alt['par']['reciprocal_phi']
        sigmas = mus + mus * mus * reciprocal_phi
        p_value = 1 - chi2(3 * (K - 1)).cdf(2 * log_likelihood)
    return p_value, log_likelihood, mus, sigmas


def init_null_NB_cov_model(model_dir):
    code = """
    data {
    int<lower=1> N;
    int<lower=1> P;
    int<lower=0> y[N];
    real<lower=0> mu_raw;
    matrix[N,P] x;  // New covariate matrix with P columns
    }
    parameters {
	real<lower=0, upper=1> theta;
	real<lower=0> mu;
	real<lower=1e-4> reciprocal_phi;
	vector[P] beta;
    }
    model {
    vector[N] xb;


    mu ~ normal(mu_raw, sqrt(mu_raw/10) + 1e-4);
    reciprocal_phi ~ normal(0,1);

    beta[1]~normal(0,sqrt(mu_raw) + 1e-4);
    xb=x*beta;

    for (n in 1:N) {
        real mu_pos;
        if ((mu+xb[n])<=0){
        mu_pos=0+1e-4;
        }
        else{
        mu_pos=mu+xb[n];
        }
        if (y[n] == 0) {
                target += log_sum_exp(bernoulli_lpmf(1 | theta),bernoulli_lpmf(0 | theta) + neg_binomial_2_lpmf(y[n] | mu_pos, 1./sqrt(reciprocal_phi)));

        } else {
                target += bernoulli_lpmf(0 | theta) + neg_binomial_2_lpmf(y[n] | mu_pos, 1./sqrt(reciprocal_phi));
	    }
	}
    }

    """
    file = f'{model_dir}/null_NB_model.cov.pkl'
    extra_compile_args = ['-pthread', '-DSTAN_THREADS']
    model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
    with open(file, 'wb') as f:
        pickle.dump(model, f)

def init_alt_NB_cov_model(model_dir):
    code = """
    data {
        int<lower=1> N;
    int<lower=1> P;
        int<lower=1> K;
        int<lower=0> y[N];
        int<lower=0> z[N, K];
        vector<lower=0>[K] mu_raw;
    matrix[N,P] x;  // New covariate matrix with P columns
    }
    parameters {
        real<lower=0, upper=1> theta[K];
        vector<lower=0>[K] mu;
        vector<lower=1e-4>[K] reciprocal_phi;
    matrix[P,K] beta;
    }
    model {
    matrix[N,K] xb;

    print(x);print(beta);
    beta[1]~normal(0,sqrt(mu_raw) + 1e-4);
    for (p in 2:P){
        beta[p]~normal(0,sqrt(mu_raw) + 1e-4);
       }

        mu ~ normal(mu_raw, sqrt(mu_raw/10)+1e-4);
        reciprocal_phi ~ normal(0, 1);
	xb=x*beta;
        for (n in 1:N) {
            vector[K] lps;
            if (y[n] == 0){
                for (k in 1:K){
                    real mu_pos;
                    if ((mu[k]+xb[n,k])<=0){
                    mu_pos=0+1e-4;
                    }
                    else{
                    mu_pos=mu[k]+xb[n,k];
                    }
                    print(y[n]," ",mu_pos);
                    lps[k] = log_sum_exp(bernoulli_lpmf(1 | theta[k]), bernoulli_lpmf(0 | theta[k]) + neg_binomial_2_lpmf(y[n] | mu_pos, 1./sqrt(reciprocal_phi[k])));
                    lps[k] *= z[n][k];
                }
            }
            else{
                for (k in 1:K) {
                    real mu_pos;
                    if ((mu[k]+xb[n,k])<=0){
                    mu_pos=0+1e-4;
                    }
                    else{
                    mu_pos=mu[k]+xb[n,k];
                    }
                    lps[k] = bernoulli_lpmf(0 | theta[k]) + neg_binomial_2_lpmf(y[n] | mu_pos, 1./sqrt(reciprocal_phi[k]));
                    lps[k] *= z[n][k];
                }
            }
            target += lps;
        }
    }

    """
    file = f'{model_dir}/alt_NB_model.cov.pkl'
    extra_compile_args = ['-pthread', '-DSTAN_THREADS']
    model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
    with open(file, 'wb') as f:
        pickle.dump(model, f)


def init_null_NB_model(model_dir):
    code = """
    data {
        int<lower=1> N;
        int<lower=0> y[N];
        real<lower=0> mu_raw;
    }
    parameters {
        real<lower=0, upper=1> theta;
        real<lower=0> mu;
        real<lower=1e-4> reciprocal_phi;
    }
    model {
        mu ~ normal(mu_raw, sqrt(mu_raw/10)+1e-4);
        reciprocal_phi ~ cauchy(0., 5);
        for (n in 1:N) {
            if (y[n] == 0){
                target += log_sum_exp(bernoulli_lpmf(1 | theta),
                                         bernoulli_lpmf(0 | theta)
                                         + neg_binomial_2_lpmf(y[n] | mu+1e-4, 1./reciprocal_phi));
            }
            else{
                target += bernoulli_lpmf(0 | theta) + neg_binomial_2_lpmf(y[n] | mu+1e-4, 1./reciprocal_phi);
            }
        }
    }
    """
    file = f'{model_dir}/null_NB_model.pkl'
    extra_compile_args = ['-pthread', '-DSTAN_THREADS']
    model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
    with open(file, 'wb') as f:
        pickle.dump(model, f)


def init_alt_NB_model(model_dir):
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
        vector<lower=1e-4>[K] reciprocal_phi;
    }
    model {
        mu ~ normal(mu_raw, sqrt(mu_raw/10)+1e-4);
        reciprocal_phi ~ cauchy(0., 5);
        for (n in 1:N) {
            vector[K] lps;
            if (y[n] == 0){
                for (k in 1:K){
                    lps[k] = log_sum_exp(bernoulli_lpmf(1 | theta[k]),
                                         bernoulli_lpmf(0 | theta[k])
                                         + neg_binomial_2_lpmf(y[n] | mu[k]+1e-4, 1./reciprocal_phi[k]));
                    lps[k] *= z[n][k];
                }
            }
            else{
                for (k in 1:K) {
                    lps[k] = bernoulli_lpmf(0 | theta[k]) + neg_binomial_2_lpmf(y[n] | mu[k]+1e-4, 1./reciprocal_phi[k]);
                    lps[k] *= z[n][k];
                }
            }
            target += lps;
        }
    }
    """
    file = f'{model_dir}/alt_NB_model.pkl'
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


def get_splice_site_groups(intron_coords):
    group_dict = defaultdict(list)
    for _chr, strand, i_start, i_end in intron_coords:
        group_dict[(i_start, 'i')].append((_chr, strand, i_start, i_end))
        group_dict[(i_end, 'o')].append((_chr, strand, i_start, i_end))

    return group_dict


## Dirichlet Multinomial model
def DM_model(df, index_df, conditions, confounders, model_dir, num_workers=4, error_rate=0.05,
            method='fdr_bh', batch_size=1000, group_filter=0, aggressive_mode=False):
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
    for _chr in chrs:
        strands = list(_index_df.loc[_chr].index.unique(level='strand'))
        for strand in strands:
            group_dict = get_splice_site_groups(_index_df.loc[_chr].loc[strand]['index'].tolist())
            for group, intron_coords in group_dict.items():
                if len(intron_coords) > 1:
                    group_df = _df.loc[intron_coords]
                    if any(group_df.sum(axis=0) >= group_filter):
                        ys.append(group_df.values.T.astype(int))
                        groups.append((f"g{i+1:06d}", (_chr, strand, group[0], group[1])))
                        coords.append(intron_coords)
                        if i > 0 and i % batch_size == 0:
                            delayed_results.append(delayed(batch_run_DM_model)(ys, conditions, confounders,model_dir, aggressive_mode))
                            groups_batches.append(groups)
                            coords_batches.append(coords)
                            groups = []
                            coords = []
                            ys = []
                        i += 1

    if len(ys) > 0:
        delayed_results.append(delayed(batch_run_DM_model)(ys, conditions, confounders,model_dir, aggressive_mode))
        groups_batches.append(groups)
        coords_batches.append(coords)

    results_batches = list(compute(*delayed_results, traverse=False, num_workers=num_workers, scheduler="processes"))

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


def batch_run_DM_model(ys, conditions, confounders, model_dir, aggressive_mode):
    null_model = None
    #null_model = pickle.load(open(Path(model_dir) / 'null_DM_model.pkl', 'rb'))
    alt_model = pickle.load(open(Path(model_dir) / 'DM_cov_model.pkl', 'rb'))
    results = []
    for y in ys:
        results.append(run_DM_model(y, conditions, confounders, null_model, alt_model, aggressive_mode))

    return results


def run_DM_model(y, conditions, confounders, null_model, alt_model, aggressive_mode):
    N, M = y.shape
    z = conditions
    K = z.shape[1]
    x_null=confounders.drop(['condition'],axis=1)
    x_alt=confounders
    if aggressive_mode:
        print("aggressive_mode to be implemented in DM model, aggressive_mode has not been enabled.")
        #conc = np.sum(np.mean(y, axis=0))
        #conc_raw, conc_std = conc, np.sqrt(conc/10) + 1e-4

    null_data_dict = {'N': N, 'M': M, 'y': y, 'conc_shape': 1.0001, 'conc_rate': 1e-4, 'P': len(x_null.columns), 'x': x_null}

    alt_data_dict = {'N': N, 'M': M, 'y': y, 'conc_shape': 1.0001, 'conc_rate': 1e-4, 'P': len(x_alt.columns), 'x': x_alt}

    max_optim_n=10
    with suppress_stdout_stderr():
        i = 0
        while i < max_optim_n:
            try:
                fit_null = alt_model.optimizing(data=null_data_dict, as_vector=False)
                fit_alt = alt_model.optimizing(data=alt_data_dict, as_vector=False)
            except RuntimeError:
                i += 1
                continue
            break

    if i == max_optim_n:
        return None, None, None
    else:
        beta=(fit_alt['par']['beta_raw']-1/len(pandas.DataFrame(fit_alt['par']['beta_raw']).columns)).T * fit_alt['par']['beta_scale']
        beta_T=beta.T
        def normalize(a):
            return a/sum(a)
        def softmax(a,normalize):
            return normalize(np.exp(a))
        def to_psi(b,conc,normalize,softmax):
            return normalize(softmax(b,normalize)*conc)
        null_psi=to_psi(beta_T[0],fit_alt['par']['conc'],normalize,softmax).tolist()
        alt_psi=to_psi(beta_T[0]+beta_T[1],fit_alt['par']['conc'],normalize,softmax).tolist()
        log_likelihood = fit_alt['value'] - fit_null['value']
        psis = np.array([alt_psi,null_psi]).T.tolist()
        p_value = 1 - chi2(M * (K - 1)).cdf(2 * (log_likelihood))
        return p_value, log_likelihood, psis


def init_DM_cov_model(model_dir):
    code = """
    data {
      int<lower=0> N; // sample size
      int<lower=0> P; // number of covariates
      int<lower=0> M; // number of classes
      vector[P] x[N]; // covariates
      vector[M] y[N]; // counts
      real<lower=0> conc_shape; // concentration shape
      real<lower=0> conc_rate; // concentration rate
    }
    parameters {
      simplex[M] beta_raw[P]; 
      real beta_scale[P];
      real<lower=0> conc[M]; // concentration parameter
    }

    model {
      matrix[M,P] beta;
      for (k in 1:M)
	for (p in 1:P)
	  beta[k,p] = beta_scale[p] * (beta_raw[p][k] - 1.0 / M);

      conc ~ gamma(conc_shape, conc_rate);
      for (n in 1:N) {
	vector[M] a;
	real suma;
	vector[M] aPlusY;
	vector[M] lGaPlusY;
	vector[M] lGaA ;
	vector[M] s;
	s = softmax(beta * x[n]);
	for (k in 1:M)
	  a[k] = conc[k] * s[k];
	// explicit construction of multinomial dirichlet
	// y ~ multinomial_dirichlet( conc * softmax(beta * x[n]) )
	suma = sum(a);
	aPlusY = a + y[n];
	for (k in 1:M) {
	  lGaPlusY[k] = lgamma(aPlusY[k]);
	  lGaA[k] = lgamma(a[k]);
	}
	target += lgamma(suma)+sum(lGaPlusY)-lgamma(suma+sum(y[n]))-sum(lGaA);
      }
    }
    """
    file = f'{model_dir}/DM_cov_model.pkl'
    extra_compile_args = ['-pthread', '-DSTAN_THREADS']
    model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
    with open(file, 'wb') as f:
        pickle.dump(model, f)

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
        real<lower=0> conc_mu;
        real<lower=0> conc_std;
    }
    parameters {
        simplex[M] alpha;
        real<lower=0> conc;
    }
    model {
        conc ~ normal(conc_mu, conc_std);
        for (n in 1:N) {
            target += dirichlet_multinomial_lpmf(y[n] | conc * alpha);
        }
    }
    """
    file = f'{model_dir}/null_DM_model.pkl'
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
        real<lower=0> conc_mu;
        real<lower=0> conc_std;
    }
    parameters {
        simplex[M] alpha[K];
        real<lower=0> conc;
    }
    model {
        conc ~ normal(conc_mu, conc_std);
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
    file = f'{model_dir}/alt_DM_model.pkl'
    extra_compile_args = ['-pthread', '-DSTAN_THREADS']
    model = pystan.StanModel(model_code=code, extra_compile_args=extra_compile_args)
    with open(file, 'wb') as f:
        pickle.dump(model, f)
