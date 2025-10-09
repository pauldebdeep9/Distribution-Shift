# ---- crash guards (must be before ANY other imports) ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # Apple Accelerate
os.environ.setdefault("PYTHONMALLOC", "debug")
os.environ.setdefault("MallocNanoZone", "0")
import faulthandler; faulthandler.enable()
# ---- end crash guards ----

# rs_mcc_only.py
# McCormick (McC) DRO procurement optimization — CVXPY MILP only
# No Mosek, no CPP, no SAA.

import numpy as np
import pandas as pd
import math
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product
import time

# External OOS evaluator (your module)
from oos_analys_RS import OOS_analys

warnings.filterwarnings("ignore")

# === CLI ===
import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regimes", choices=["1","2","both"], default="both",
                   help="1=no-regime (pooled), 2=regime-switching, both=run both & compare")
    p.add_argument("--tag", default="", help="suffix for output files/logs")
    return p.parse_args()

from typing import Tuple

def _oos_as_dataframe(cost_obj) -> pd.DataFrame:
    """
    Best-effort converter: expect attributes like:
      total_cost, fixed_order_cost, purchase_cost, inv_cost, backlog_cost
    that are 1D arrays/Series over OoS samples. Fallbacks handle scalars.
    """
    def _series(x, name):
        try:
            return pd.Series(x, name=name)
        except Exception:
            return pd.Series([float(x)], name=name)

    cols = []
    cols.append(_series(getattr(cost_obj, "total_cost", []), "total"))
    cols.append(_series(getattr(cost_obj, "fixed_order_cost", []), "fixed_order"))
    cols.append(_series(getattr(cost_obj, "purchase_cost", []), "purchase"))
    cols.append(_series(getattr(cost_obj, "inv_cost", []), "inventory"))
    cols.append(_series(getattr(cost_obj, "backlog_cost", []), "backlog"))
    df = pd.concat(cols, axis=1)

    # ensure a sample_id 0..M-1
    df.insert(0, "sample_id", range(len(df)))
    return df



# ============================ HMM helper ============================

def train_hmm_with_sorted_states(data, n_components=2, random_state=42, covariance_type='full', n_iter=100):
    """
    Train a Gaussian HMM on 1-D labels; sort states by mean so indices are stable/interpretable.
    Returns:
        model (fitted and re-ordered),
        remapped_states (np.array),
        sorted_transmat,
        sorted_means,
        state_counts (np.array length n_components)
    """
    # lazy import to avoid early native-lib mixing
    from hmmlearn import hmm

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(data)
    hidden_states = model.predict(data)

    means = model.means_.flatten()
    sort_idx = np.argsort(means)
    mapping = {old: new for new, old in enumerate(sort_idx)}
    remapped = np.array([mapping[s] for s in hidden_states])

    state_counts = np.bincount(remapped, minlength=n_components)
    sorted_transmat = model.transmat_[sort_idx][:, sort_idx]
    sorted_means = model.means_[sort_idx]
    sorted_covars = model.covars_[sort_idx]

    model.transmat_ = sorted_transmat
    model.means_ = sorted_means
    model.covars_ = sorted_covars
    return model, remapped, sorted_transmat, sorted_means, state_counts


def save_transition_matrix_analysis(transmat, means, counts, N, K, filename_prefix="transition_analysis"):
    """
    Save transition matrix analysis to files for further examination.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transition matrix
    transmat_df = pd.DataFrame(transmat, 
                              columns=[f"To_State_{i}" for i in range(K)],
                              index=[f"From_State_{i}" for i in range(K)])
    transmat_df.to_csv(f"{filename_prefix}_N{N}_K{K}_{timestamp}.csv")
    
    # Save summary statistics
    summary = {
        "N": N,
        "K": K,
        "State_Means": means.flatten().tolist() if hasattr(means, 'flatten') else means,
        "State_Counts": counts.tolist() if hasattr(counts, 'tolist') else counts,
        "Stationary_Distribution": None
    }
    
    try:
        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(transmat.T)
        stationary = eigenvecs[:, 0].real
        stationary = stationary / np.sum(stationary)
        summary["Stationary_Distribution"] = stationary.tolist()
    except:
        summary["Stationary_Distribution"] = "Could not compute"
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{filename_prefix}_summary_N{N}_K{K}_{timestamp}.csv", index=False)
    
    print(f"Transition matrix analysis saved to files with prefix: {filename_prefix}_N{N}_K{K}_{timestamp}")
    return transmat_df, summary_df


# ============================ demand generators ============================

from scipy.special import gamma as sp_gamma
from scipy.optimize import fsolve

def get_gaussian_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean_vector = np.full((time_horizon,), mean)
    covariance_matrix = np.diag(np.full((time_horizon,), std_dev**2))
    cov_value = rho * std_dev**2
    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value
    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    samples = np.round(samples).astype(int).T
    return pd.DataFrame(samples)

def get_gamma_demand(time_horizon, mean, std_dev, M, seed=None):
    if seed is not None:
        np.random.seed(seed)
    k = mean**2 / std_dev**2
    theta = std_dev**2 / mean
    m = time_horizon * M
    samples = np.random.gamma(shape=k, scale=theta, size=m)
    samples = np.round(samples).astype(int).reshape((time_horizon, M))
    return pd.DataFrame(samples)

def get_lognormal_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)
    normal_mean = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
    normal_std_dev = np.sqrt(np.log(1 + (std_dev**2 / mean**2)))
    mean_vector = np.full((time_horizon,), normal_mean)
    covariance_matrix = np.diag(np.full((time_horizon,), normal_std_dev**2))
    cov_value = rho * normal_std_dev**2
    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value
    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    samples = np.round(np.exp(samples)).astype(int).T
    return pd.DataFrame(samples)

def get_weibull_demand(time_horizon, mean, std_dev, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def weibull_params(mean_, std_):
        def equations(vars_):
            k_, lam_ = vars_
            mean_eq = lam_ * sp_gamma(1 + 1/k_) - mean_
            std_eq = lam_ * np.sqrt(sp_gamma(1 + 2/k_) - sp_gamma(1 + 1/k_)**2) - std_
            return [mean_eq, std_eq]
        return fsolve(equations, (0.5, mean_))

    k, lam = weibull_params(mean, std_dev)
    samples = np.random.weibull(k, size=(time_horizon, M)) * lam
    samples = np.round(samples).astype(int)
    return pd.DataFrame(samples)

def get_mixture_demand(distribution_list, time_horizon, M, seed=None, M_max=20000):
    if seed is not None:
        np.random.seed(seed)

    all_samples = []
    all_labels = []
    for i, dist_info in enumerate(distribution_list):
        dist_type = dist_info['type'].lower()
        weight = dist_info['weight']
        params = dist_info['params']
        sample_size_max = int(round(weight * M_max))
        local_seed = (seed or 0) + i * 99991

        if dist_type == 'gaussian':
            samples = get_gaussian_demand(time_horizon, params['mean'], params['std_dev'], params.get('rho', 0), sample_size_max, local_seed)
        elif dist_type == 'gamma':
            samples = get_gamma_demand(time_horizon, params['mean'], params['std_dev'], sample_size_max, local_seed)
        elif dist_type == 'lognormal':
            samples = get_lognormal_demand(time_horizon, params['mean'], params['std_dev'], params.get('rho', 0), sample_size_max, local_seed)
        elif dist_type == 'weibull':
            samples = get_weibull_demand(time_horizon, params['mean'], params['std_dev'], sample_size_max, local_seed)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        sample_size = int(round(weight * M))
        samples = samples.iloc[:, :sample_size]
        all_samples.append(samples)
        all_labels.extend([i]*sample_size)

    combined = pd.concat(all_samples, axis=1).T
    combined['label'] = all_labels
    shuffled = combined.sample(frac=1, random_state=seed).reset_index(drop=True).iloc[:M, :]
    labels = shuffled['label']
    samples_df = shuffled.drop(columns='label').T
    samples_df.columns = range(samples_df.shape[1])
    return samples_df, labels


# ============================ Model class (McC only) ============================

class Models:
    def __init__(self, h, b, I_0, B_0, R, input_parameters_file, dist, input_demand, N):
        data_price    = pd.read_excel(input_parameters_file, sheet_name='price')
        data_supplier = pd.read_excel(input_parameters_file, sheet_name='supplier')
        data_capacity = pd.read_excel(input_parameters_file, sheet_name='capacity')

        self.h = h
        self.b = b
        self.I_0 = I_0
        self.B_0 = B_0
        self.N = N
        self.Nlist = list(range(N))
        self.R = R
        self.dist = dist
        self.demand = input_demand  # dict: state -> DataFrame (time x Nk[state])
        self.price_df = data_price

        self.time = range(next(iter(input_demand.values())).shape[0])  # rows of any state's DF
        self.supplier, self.order_cost, self.lead_time, self.quality_level = self.get_suppliers(data_supplier)
        self.prices, self.capacities = self.get_time_suppliers(data_price, data_capacity)

        self.t_supplier = [(t, s) for t in self.time for s in self.supplier]
        self.t_supplier_n = [(t, s, n) for t in self.time for s in self.supplier for n in self.Nlist]

    @staticmethod
    def get_structure(*args):
        if len(args) == 2:
            return [(a, b) for a in args[0] for b in args[1]]
        if len(args) == 3:
            return [(a, b, c) for a in args[0] for b in args[1] for c in args[2]]
        if len(args) == 4:
            return [(a, b, c, d) for a in args[0] for b in args[1] for c in args[2] for d in args[3]]
        raise ValueError("get_structure supports up to 4 iterable args")

    @staticmethod
    def _multidict_like(multi_temp):
        """
        Mimic gurobipy.multidict: input {key: [v1, v2, v3]}
        return (keys_list, dict1, dict2, dict3)
        """
        keys = list(multi_temp.keys())
        d1, d2, d3 = {}, {}, {}
        for k, vals in multi_temp.items():
            d1[k], d2[k], d3[k] = vals
        return keys, d1, d2, d3

    def get_suppliers(self, data_supplier):
        supplier = data_supplier['supplier'].values
        order_cost = data_supplier['order_cost'].values
        lead_time = data_supplier['lead_time'].values
        quality_level = data_supplier['quality_level'].values
        multi_temp = {}
        for i in range(len(supplier)):
            multi_temp[supplier[i]] = [float(order_cost[i]), float(lead_time[i]), float(quality_level[i])]
        return self._multidict_like(multi_temp)

    def get_time_suppliers(self, data_price, data_capacity):
        price_sn, capacity_sn = [0], [0]
        for i in range(1, len(self.supplier)+1):
            price_sn.append(data_price['s'+str(i)].values)
            capacity_sn.append(data_capacity['s'+str(i)].values)
        prices = {}
        capacities = {}
        for t in self.time:
            for s in self.supplier:
                n = int(s[1:])
                prices[(t, s)] = float(price_sn[n][t])
                capacities[(t, s)] = float(capacity_sn[n][t])
        return prices, capacities

    # ================= McC MILP in CVXPY (exact translation of your Gurobi model) =================
    def optimize_McC_rs(self, K, epsilon, r, Nk, solver_preference=None, verbose=False):
        """
        CVXPY MILP version of your McCormick (McC) DRO model.
        Args:
            K: number of regimes
            epsilon: dict {state -> epsilon_k}
            r: array-like of length K (weights/probabilities)
            Nk: array-like of length K (sample counts for each regime)
        Returns:
            (objective_value, df_solution)
        """
        # lazy import to isolate solver backends
        import cvxpy as cp

        T_idx = list(self.time)
        S = list(self.supplier)
        T = T_idx[-1]

        P_I = self.I_0 - self.B_0
        b = self.b
        h = self.h
        Mbig = 999_999.0

        # demand bounds per (k,n)
        dU, dL = {}, {}
        for k in range(K):
            dU[k], dL[k] = {}, {}
            for n in range(Nk[k]):
                col = self.demand[k].iloc[:, n]
                dU[k][n] = float(col.max())
                dL[k][n] = float(col.min())

        # decision variables
        Q     = {(t, s): cp.Variable(nonneg=True,  name=f"order_quantity[{t},{s}]")   for t in T_idx for s in S}
        theta = {(t, s): cp.Variable(nonneg=True,  name=f"arrive_quantity[{t},{s}]")  for t in T_idx for s in S}
        Y     = {(t, s): cp.Variable(boolean=True, name=f"if_make_order_arrive[{t},{s}]") for t in T_idx for s in S}

        alpha = {(k, n): cp.Variable(name=f"alpha[{k},{n}]") for k in range(K) for n in range(Nk[k])}
        beta  = {k: cp.Variable(nonneg=True, name=f"beta[{k}]") for k in range(K)}

        delta = {(k,n,t): cp.Variable(nonneg=True, name=f"delta[{k},{n},{t}]") for k in range(K) for n in range(Nk[k]) for t in T_idx}
        sigma = {(k,n,t): cp.Variable(nonneg=True, name=f"sigma[{k},{n},{t}]") for k in range(K) for n in range(Nk[k]) for t in T_idx}
        gamma = {(k,n,t): cp.Variable(nonneg=True, name=f"gamma[{k},{n},{t}]") for k in range(K) for n in range(Nk[k]) for t in T_idx}
        tau   = {(k,n,t): cp.Variable(nonneg=True, name=f"tau[{k},{n},{t}]")   for k in range(K) for n in range(Nk[k]) for t in T_idx}
        phi   = {(k,n,t): cp.Variable(nonneg=True, name=f"phi[{k},{n},{t}]")   for k in range(K) for n in range(Nk[k]) for t in T_idx}
        xi    = {(k,n,t): cp.Variable(nonneg=True, name=f"xi[{k},{n},{t}]")    for k in range(K) for n in range(Nk[k]) for t in T_idx}
        zeta  = {(k,n,t): cp.Variable(nonneg=True, name=f"zeta[{k},{n},{t}]")  for k in range(K) for n in range(Nk[k]) for t in T_idx}
        varphi= {(k,n,t): cp.Variable(nonneg=True, name=f"varphi[{k},{n},{t}]")for k in range(K) for n in range(Nk[k]) for t in T_idx}
        eta   = {(k,n,t): cp.Variable(nonneg=True, name=f"eta[{k},{n},{t}]")   for k in range(K) for n in range(Nk[k]) for t in T_idx}
        ta    = {(k,n)  : cp.Variable(nonneg=True, name=f"ta[{k},{n}]")        for k in range(K) for n in range(Nk[k])}

        cons = []

        # arrival linkage: theta[tp,s] = sum_t Q[t,s] where t + L_s == tp
        for s in S:
            Lt = int(round(self.lead_time[s]))
            for tp in T_idx:
                cons.append(theta[tp, s] == cp.sum([Q[t, s] for t in T_idx if t + Lt == tp]))

        # big-M and capacity
        for t in T_idx:
            for s in S:
                cons += [
                    Q[t, s] <= Mbig * Y[t, s],
                    Q[t, s] <= self.capacities[(t, s)],
                    Y[t, s] >= 0,
                    Y[t, s] <= 1
                ]

        # McC DRO constraints
        for k in range(K):
            for n in range(Nk[k]):
                cons.append(ta[k, n] <= beta[k])

            for n in range(Nk[k]):
                for t in T_idx:
                    cons.append(delta[k, n, t] + sigma[k, n, t] - ta[k, n] <= 0)

            for n in range(Nk[k]):
                # master inequality bounding alpha[k,n]
                terms = []
                terms.append((b + h) * cp.sum(cp.hstack([tau[k, n, t] for t in T_idx if t < T])))
                terms.append((T + 1) * h * P_I)
                for t in T_idx:
                    dnt = float(self.demand[k].iloc[t, n])
                    terms.append((delta[k, n, t] - sigma[k, n, t]) * dnt)
                    terms.append((T - t + 1) * h * cp.sum(cp.hstack([theta[t, s] for s in S])))
                    terms.append(xi[k, n, t]    * (T - t + 1) * (b + h) * dU[k][n])
                    terms.append(- zeta[k, n, t]* (T - t + 1) * (b + h) * dL[k][n])
                    terms.append(eta[k, n, t]   * (T - t + 1) * (b + h))
                cons.append(cp.sum(cp.hstack(terms)) <= alpha[k, n])

                for t in T_idx:
                    cons.append(1 + phi[k, n, t] + xi[k, n, t] - zeta[k, n, t] - varphi[k, n, t] <= 0)

                for t in T_idx:
                    cons.append(
                        sigma[k, n, t] - delta[k, n, t]
                        + (zeta[k, n, t] - xi[k, n, t]) * (T - t + 1) * (h + b)
                        - (T - t + 1) * h
                        <= 0
                    )

                for t in T_idx:
                    sum_theta_t = cp.sum(cp.hstack([theta[t, s] for s in S]))
                    if t == 0:
                        cons.append(
                            gamma[k, n, t] - tau[k, n, t]
                            - phi[k, n, t] * dL[k][n] - xi[k, n, t] * dU[k][n]
                            + zeta[k, n, t] * dL[k][n] + varphi[k, n, t] * dU[k][n]
                            - eta[k, n, t] - sum_theta_t - P_I
                            <= 0
                        )
                    elif t == T:
                        cons.append(
                            tau[k, n, t - 1] - gamma[k, n, t - 1]
                            - phi[k, n, t] * dL[k][n] - xi[k, n, t] * dU[k][n]
                            + zeta[k, n, t] * dL[k][n] + varphi[k, n, t] * dU[k][n]
                            - eta[k, n, t] - sum_theta_t
                            <= 0
                        )
                    else:
                        cons.append(
                            gamma[k, n, t] - tau[k, n, t] + tau[k, n, t - 1] - gamma[k, n, t - 1]
                            - phi[k, n, t] * dL[k][n] - xi[k, n, t] * dU[k][n]
                            + zeta[k, n, t] * dL[k][n] + varphi[k, n, t] * dU[k][n]
                            - eta[k, n, t] - sum_theta_t
                            <= 0
                        )

        # objective
        fixed_order = np.sum([ self.order_cost[s] * 1.0 for s in S ])  # original fixed term uses Y; kept below in cvxpy form
        fixed_order = (lambda: None)  # placeholder to keep structure; actual in cvxpy below
        fixed_order = sum([ self.order_cost[s] * Y[t, s] for t in T_idx for s in S ])
        purchase    = sum([ self.prices[(t, s)] * Q[t, s] for t in T_idx for s in S ])
        dro_term    = sum([ (r[k] * sum([alpha[k, n] for n in range(Nk[k])]) / Nk[k]
                             + r[k] * beta[k] * epsilon[k]) for k in range(K) ])
        obj = fixed_order + purchase + dro_term

        prob = cp.Problem(cp.Minimize(obj), cons)

        # solver order (MIP-capable)
        order = []
        if solver_preference is not None:
            order.append(solver_preference)
        order += [cp.SCIPY, cp.GUROBI, cp.CBC, cp.GLPK_MI, cp.SCIP, cp.ECOS_BB]

        status = None
        for s in order:
            if s not in cp.installed_solvers():
                continue
            try:
                prob.solve(solver=s, verbose=verbose)
                status = prob.status
                if status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                continue
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY failed; last status: {status}; installed={cp.installed_solvers()}")

        # export solution in a simple DataFrame
        rows = []
        for (t, s), v in Q.items():
            rows.append({"variable_name": f"order_quantity[{t},{s}]", "value": float(v.value)})
        for (t, s), v in theta.items():
            rows.append({"variable_name": f"arrive_quantity[{t},{s}]", "value": float(v.value)})
        for (t, s), v in Y.items():
            rows.append({"variable_name": f"if_make_order_arrive[{t},{s}]", "value": float(v.value)})
        df_result = pd.DataFrame(rows)
        return float(prob.value), df_result


def run_experiment_for_K(K, Demand_samples, Label_samples, common_cfg):
    (h,b,I_0,B_0,R,input_parameters_file,planning_horizon,k_fold,oos_size,seed,input_sample_no) = common_cfg
    oos_analys = OOS_analys(h, b, I_0, B_0, input_parameters_file)
    all_rows = []

    for N in input_sample_no:
        input_demand = Demand_samples.iloc[:, :N]
        labels = Label_samples[:N]
        data = np.array(input_demand).T  # (N, T)

        # --- split (RS) or pool (no-RS) ---
        if K > 1:
            Labels = labels.values.reshape(len(labels), 1)
            model, _, _, _, _ = train_hmm_with_sorted_states(Labels, n_components=K,
                                                             random_state=seed, covariance_type='full', n_iter=100)
            state_samples_dict = {k: [] for k in range(K)}
            for i, st in enumerate(labels):
                state_samples_dict[int(st)].append(data[i])
            state_samples_df = {k: pd.DataFrame(np.array(v).T if len(v)>0 else np.zeros((planning_horizon,0)))
                                for k,v in state_samples_dict.items()}
            for k in range(K):
                state_samples_df[k].to_csv(f"state_{k}_samples.csv", index=False)
            r_last = model.transmat_[int(labels.iloc[-1])]
        else:
            state_samples_df = {0: pd.DataFrame(data.T)}
            r_last = np.array([1.0])

        # --- CV epsilon per state (works for K=1 too) ---
        list_epsilon = [0, 30, 60]
        from itertools import product
        grid = list(product(*([list_epsilon]*K)))
        best_eps_vecs = []
        num_fold = N // k_fold

        for fold in range(1, k_fold):
            all_cols = list(range(input_demand.shape[1]))
            selected_cols = [i for i in all_cols if i < fold*num_fold or i >= (fold+1)*num_fold]
            train_demand = input_demand.iloc[:, selected_cols]
            train_labels = labels[selected_cols].values.reshape(-1, 1)
            if K > 1:
                model_cv, *_ = train_hmm_with_sorted_states(train_labels, n_components=K,
                                                            random_state=seed, covariance_type='full', n_iter=100)
            # build state-indexed train frames
            train_input_demand = {}
            if K > 1:
                for st in range(K):
                    mask = (train_labels.flatten() == st)
                    train_input_demand[st] = train_demand.loc[:, mask].reset_index(drop=True)
            else:
                train_input_demand[0] = train_demand.reset_index(drop=True)

            solve = Models(h, b, I_0, B_0, R, input_parameters_file, 'mix', train_input_demand, train_demand.shape[1])

            min_cost = 1e18
            best_eps = None
            CV_slice = input_demand.iloc[:, fold*num_fold:(fold+1)*num_fold]

            for eps_tuple in grid:
                eps_dict = {st: eps_tuple[st] for st in range(K)}
                fold_costs = []
                for j in range(num_fold):
                    if K > 1:
                        idx = min(N-1, fold*num_fold + j - 1)
                        last_label = int(labels.iloc[idx]) if idx >= 0 else 0
                        r_vec = model_cv.transmat_[last_label]
                    else:
                        r_vec = np.array([1.0])
                    Nk_train = [max(1, train_input_demand.get(st, pd.DataFrame()).shape[1]) for st in range(K)]
                    obj, solution = solve.optimize_McC_rs(K, eps_dict, r_vec, Nk_train,
                                                          solver_preference=None, verbose=False)
                    cost, _ = oos_analys.cal_out_of_sample(solution, CV_slice.iloc[:, j])
                    fold_costs.append(cost.total_cost.mean())
                avg_cost = float(np.mean(fold_costs))
                if avg_cost < min_cost:
                    min_cost, best_eps = avg_cost, eps_dict
            best_eps_vecs.append(best_eps)

        eps_final = {st: np.mean([vec[st] for vec in best_eps_vecs]) for st in range(K)}

        # --- final solve on full input ---
        if K > 1:
            input_demand_regime = {st: pd.read_csv(f"state_{st}_samples.csv") for st in range(K)}
        else:
            input_demand_regime = state_samples_df
        Nk_final = [max(1, input_demand_regime.get(st, pd.DataFrame()).shape[1]) for st in range(K)]
        solve_full = Models(h, b, I_0, B_0, R, input_parameters_file, 'mix', input_demand_regime, N)
        obj, solution = solve_full.optimize_McC_rs(K, eps_final, r_last, Nk_final,
                                                   solver_preference=None, verbose=False)

        # --- OoS mixtures ---
        for weight in [0.3, 0.5, 0.7]:
            oos_demands_mix, _ = get_mixture_demand(
                distribution_list=[
                    {'type': 'gaussian', 'weight': weight,   'params': {'mean': 1800, 'std_dev': 500, 'rho': 0}},
                    {'type': 'gamma',    'weight': 1-weight, 'params': {'mean': 2000, 'std_dev': 500}}
                ],
                time_horizon=planning_horizon, M=common_cfg[8], seed=common_cfg[9]
            )
            oos_cost, _ = oos_analys.cal_out_of_sample(solution, oos_demands_mix)
            all_rows.append({
                "mode": ("RS" if K>1 else "noRS"),
                "K": K, "N": N, "oos_weight": weight, "obj": obj, "epsilon": eps_final,
                "oos_mean": oos_cost.total_cost.mean(),
                "order": oos_cost.fixed_order_cost.mean(),
                "purchase": oos_cost.purchase_cost.mean(),
                "oos_inv": oos_cost.inv_cost.mean(),
                "oos_backlog": oos_cost.backlog_cost.mean(),
                "oos_std": oos_cost.total_cost.std(),
            })
    return pd.DataFrame(all_rows)


# ============================ experiment driver (McC only) ============================

if __name__ == "__main__":
    input_parameters_file = 'input_parameters.xlsx'
    h, b, I_0, B_0, R = 5, 20, 1800, 0, 0

    k_fold = 5
    oos_size = 100

    planning_horizon = 8
    rho = 0

    seed = 25
    random_state = seed

    input_sample_no = [10, 20, 40]  # adjust as you like
    oos_analys = OOS_analys(h, b, I_0, B_0, input_parameters_file)

    # generate input samples (mixture) ONCE and reuse for both K=1 and K=2
    Demand_samples, Label_samples = get_mixture_demand(
        distribution_list=[
            {'type': 'gaussian', 'weight': 0.5, 'params': {'mean': 1800, 'std_dev': 500, 'rho': 0}},
            {'type': 'gamma',    'weight': 0.5, 'params': {'mean': 2000, 'std_dev': 500}}
        ],
        time_horizon=planning_horizon, M=200, seed=seed
    )

    # Collect results for both modes
    results_rows = []

    # Run both modes: K=1 (noRS) and K=2 (RS)
    for K in [1, 2]:
        mode_name = "noRS" if K == 1 else "RS"

        for input_dist in ['mix']:
            for out_sample_dist in ['mix']:
                for N in input_sample_no:
                    start = time.time()

                    # slice inputs
                    input_demand = Demand_samples.iloc[:, :N]
                    labels = Label_samples.iloc[:N]
                    data = np.array(input_demand).T  # (N, T)

                    # ---------- Build per-state sample frames ----------
                    if K > 1:
                        # Train HMM on labels (1-D)
                        Labels = labels.values.reshape(len(labels), 1)
                        model, remapped_states, sorted_transmat, sorted_means, state_counts = train_hmm_with_sorted_states(
                            Labels, n_components=K, random_state=random_state, covariance_type='full', n_iter=100
                        )
                        
                        # Display the learned transition matrix
                        print(f"\n=== Learned Transition Matrix for N={N}, K={K} ===")
                        print("Transition Matrix (sorted by state means):")
                        print(f"Shape: {sorted_transmat.shape}")
                        print(sorted_transmat)
                        print(f"\nState means: {sorted_means.flatten()}")
                        print(f"State counts: {state_counts}")
                        print(f"Stationary distribution: {np.linalg.eig(sorted_transmat.T)[1][:, 0].real / np.sum(np.linalg.eig(sorted_transmat.T)[1][:, 0].real)}")
                        print("=" * 60)
                        
                        # Save transition matrix analysis
                        save_transition_matrix_analysis(sorted_transmat, sorted_means, state_counts, N, K)
                        state_samples_dict = {k: [] for k in range(K)}
                        for i, st in enumerate(labels):
                            state_samples_dict[int(st)].append(data[i])

                        state_samples_df = {}
                        state_counts = []
                        for st, samples in state_samples_dict.items():
                            samples_array = np.array(samples).T if len(samples) > 0 else np.zeros((planning_horizon, 0))
                            state_samples_df[st] = pd.DataFrame(samples_array, columns=[f"{j}" for j in range(samples_array.shape[1])])
                            state_counts.append(state_samples_df[st].shape[1])

                        # persist (your downstream expects CSVs in original flow)
                        for st, df_st in state_samples_df.items():
                            df_st.to_csv(f"state_{st}_samples.csv", index=False)

                        # next-state probabilities from last observed label
                        state_frequencies = model.transmat_[int(labels.iloc[-1])]
                    else:
                        # no-RS: single pooled state
                        state_samples_df = {0: pd.DataFrame(data.T)}
                        state_counts = [N]
                        state_frequencies = np.array([1.0])  # only one regime

                    # ---------- Cross-validate epsilon per state ----------
                    list_epsilon = [0, 30, 60]
                    # All K-dimensional epsilon combos
                    from itertools import product as _prod
                    eps_grid = list(_prod(*([list_epsilon] * K)))

                    # store best epsilon dict per fold
                    min_epsilons = [{st: 0 for st in range(K)} for _ in range(k_fold - 1)]

                    num_fold = N // k_fold
                    train_size = N - num_fold

                    for k in range(1, k_fold):
                        all_cols = list(range(input_demand.shape[1]))
                        sel_cols = [i for i in all_cols if i < k * num_fold or i >= (k + 1) * num_fold]

                        train_demand = input_demand.iloc[:, sel_cols]
                        train_demand.columns = range(train_demand.shape[1])

                        CV_input_demand = input_demand.iloc[:, k * num_fold:(k + 1) * num_fold]
                        CV_input_demand.columns = range(num_fold)

                        if K > 1:
                            train_labels = labels.iloc[sel_cols].values.reshape(-1, 1)
                            model_cv, remapped_cv, transmat_cv, means_cv, counts_cv = train_hmm_with_sorted_states(
                                train_labels, n_components=K, random_state=random_state, covariance_type='full', n_iter=100
                            )
                            
                            # Display CV transition matrix
                            print(f"\n--- CV Fold {k} Transition Matrix ---")
                            print(f"Training samples: {len(sel_cols)}")
                            print("CV Transition Matrix:")
                            print(transmat_cv)
                            print(f"CV State means: {means_cv.flatten()}")
                            print(f"CV State counts: {counts_cv}")
                            print("-" * 40)
                        # build state-indexed training frames
                        train_input_demand = {}
                        if K > 1:
                            tr_lab_flat = labels.iloc[sel_cols].values.flatten()
                            for st in range(K):
                                mask = (tr_lab_flat == st)
                                train_input_demand[st] = train_demand.loc[:, mask]
                                train_input_demand[st].columns = range(train_input_demand[st].shape[1])
                        else:
                            train_input_demand[0] = train_demand.reset_index(drop=True)

                        # solver wrapper
                        solve = Models(h, b, I_0, B_0, R, input_parameters_file, input_dist, train_input_demand, train_size)

                        # reference row if we need a fallback r
                        if K > 1:
                            idx_row = int(labels.iloc[k * num_fold - 1]) if (k * num_fold - 1) >= 0 else 0
                            tran_matrix = model_cv.transmat_[idx_row]
                        else:
                            tran_matrix = np.array([1.0])

                        best_avg, best_eps = 1e18, None

                        for eps_tuple in eps_grid:
                            eps_dict = {st: eps_tuple[st] for st in range(K)}
                            fold_costs = []
                            for j in range(num_fold):
                                # r_vec for this CV point
                                if K > 1:
                                    idx_hist = min(N - 1, k * num_fold + j - 1)
                                    r_vec = model_cv.transmat_[int(labels.iloc[idx_hist])] if idx_hist >= 0 else tran_matrix
                                else:
                                    r_vec = np.array([1.0])

                                # Nk per state (guard zero)
                                Nk_train = [train_input_demand.get(st, pd.DataFrame()).shape[1] for st in range(K)]
                                Nk_sanitized = [max(1, x) for x in Nk_train]

                                obj_cv, sol_cv = solve.optimize_McC_rs(K, eps_dict, r_vec, Nk_sanitized,
                                                                       solver_preference=None, verbose=False)

                                # OoS on validation column j
                                cost_cv, _ = oos_analys.cal_out_of_sample(sol_cv, CV_input_demand[j])
                                fold_costs.append(cost_cv.total_cost.mean())

                            avg_cost = float(np.mean(fold_costs))
                            if avg_cost < best_avg:
                                best_avg, best_eps = avg_cost, eps_dict

                        min_epsilons[k - 1] = best_eps

                    # average best eps across folds (per state)
                    eps_final = {st: np.mean([min_epsilons[i][st] for i in range(k_fold - 1)]) for st in range(K)}

                    # ---------- Final solve on full input ----------
                    if K > 1:
                        input_demand_regime = {st: pd.read_csv(f'state_{st}_samples.csv') for st in range(K)}
                    else:
                        input_demand_regime = state_samples_df

                    # Nk for final (guard zero)
                    Nk_final = [max(1, input_demand_regime.get(st, pd.DataFrame()).shape[1]) for st in range(K)]

                    solve_full = Models(h, b, I_0, B_0, R, input_parameters_file, input_dist, input_demand_regime, N)
                    obj, solution = solve_full.optimize_McC_rs(K, eps_final, state_frequencies, Nk_final,
                                                               solver_preference=None, verbose=False)

                    # ---------- OoS evaluation under different mixture weights ----------
                    for weight in [0.3, 0.5, 0.7]:
                        oos_demands_mix, _ = get_mixture_demand(
                            distribution_list=[
                                {'type': 'gaussian', 'weight': weight,   'params': {'mean': 1800, 'std_dev': 500, 'rho': 0}},
                                {'type': 'gamma',    'weight': 1 - weight, 'params': {'mean': 2000, 'std_dev': 500}}
                            ],
                            time_horizon=planning_horizon, M=oos_size, seed=seed
                        )

                        oos_cost, _ = oos_analys.cal_out_of_sample(solution, oos_demands_mix)
                        end = time.time()

                        results_rows.append({
                            "mode": mode_name,
                            "K": K,
                            "input_dist": input_dist,
                            "N": N,
                            "oos_weight": weight,
                            "obj": obj,
                            "epsilon": eps_final,
                            "oos_mean": oos_cost.total_cost.mean(),
                            "order": oos_cost.fixed_order_cost.mean(),
                            "purchase": oos_cost.purchase_cost.mean(),
                            "oos_inv": oos_cost.inv_cost.mean(),
                            "oos_backlog": oos_cost.backlog_cost.mean(),
                            "oos_std": oos_cost.total_cost.std(),
                            "seed": seed,
                            "time_min": (end - start) / 60.0
                        })

                    # (optional) progress print
                    print(f"[{mode_name}] N={N}: eps={eps_final}  state_counts={state_counts}  mean(oos) last={results_rows[-1]['oos_mean']:.3f}")

    # -------- Save combined results + comparison table --------
    if results_rows:
        res_df = pd.DataFrame(results_rows)
        res_df.to_excel('results_RS_vs_noRS.xlsx', index=False)

        # pivot to compare RS vs noRS mean OoS cost per (N, mix)
        pv = (res_df.pivot_table(index=["N", "oos_weight"], columns="mode", values="oos_mean")
              .reset_index())
        if {"RS", "noRS"}.issubset(pv.columns):
            pv["delta_abs"] = pv["noRS"] - pv["RS"]
            pv["delta_pct"] = 100.0 * pv["delta_abs"] / pv["noRS"]
            pv.to_excel("results_comparison.xlsx", index=False)

        pd.options.display.float_format = '{:.2f}'.format
        print(res_df)
        if {"RS", "noRS"}.issubset(pv.columns):
            print("\nComparison (lower is better):")
            print(pv)
    else:
        print("No results to save.")
