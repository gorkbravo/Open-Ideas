import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ----------- load FRED TFP data -----------
df = pd.read_csv('Your file path')
df.columns = ['date', 'tfp']
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
fred_annual = df.groupby('year')['tfp'].mean()  # many obs are annual already
start_year, end_year = 1955, 2019
fred_annual = fred_annual.loc[start_year:end_year]
years = fred_annual.index.values
T = len(years)
fred_values = fred_annual.values
# rebase to 2017 = 1
fred_values = fred_values / fred_annual.loc[2017]

# ----------- simulate model (quick, same parameters) -----------
np.random.seed(21)
n_nodes = 100
G = nx.watts_strogatz_graph(n_nodes, k=6, p=0.15)
W = nx.to_numpy_array(G) * 0.5
deg = (W > 0).sum(axis=1)
insider_idx = np.where(deg >= 10)[0]

beta = np.full(n_nodes, 0.10)
beta[insider_idx] = -0.05
gamma = np.full(n_nodes, 0.012)
gamma[insider_idx] = 0.018

psi = 1.0
sigma = 0.03
kappa = 0.20
lambda0 = 0.03
theta = 5.0
eta = 0.30
delta = 0.05
lambda_ICT_mult = 3.0
ICT_years = set(range(1994, 2006))
lambda_cap = 10.0
recession_years = {1957, 1958, 1960, 1961, 1969, 1970, 1973, 1974, 1975,
                   1980, 1981, 1982, 1990, 1991, 2001, 2007, 2008, 2009}

A = np.random.lognormal(mean=np.log(0.65), sigma=0.15, size=n_nodes)
meanA = np.zeros(T)
A_time = np.zeros((T, n_nodes))
meanA[0] = A.mean(); A_time[0] = A

for t in range(1, T):
    year = years[t]
    diff_pos = np.maximum(A.reshape(1, -1) - A.reshape(-1, 1), 0)
    diffusion = (W * diff_pos).sum(axis=1) * beta
    drift = psi * gamma
    macro_shock = np.random.normal(scale=sigma * 0.4)
    brownian = macro_shock * A
    idio = np.random.normal(scale=sigma, size=n_nodes) * A
    S = W @ A
    lam = lambda0 * np.maximum(0., S - theta)
    if year in ICT_years: lam *= lambda_ICT_mult
    lam = np.clip(lam, 0, lambda_cap)
    jumps = np.random.poisson(lam)
    A += (diffusion + drift) + brownian + idio + kappa * A * jumps
    if year in recession_years: A *= 0.985
    A = np.maximum(A, 1e-4)
    # update weights
    gap = np.abs(A.reshape(-1,1) - A.reshape(1,-1))
    W += eta * gap * W - delta * W
    W = np.clip(W, 0, 1.0); np.fill_diagonal(W, 0.0)
    meanA[t] = A.mean(); A_time[t] = A

# scale model mean to 2017 = 1
scale_factor = fred_values[list(years).index(2017)] / meanA[list(years).index(2017)]
meanA_scaled = meanA * scale_factor

# ------- plot overlay ---------
plt.figure(figsize=(8,4))
plt.plot(years, fred_values, label='FRED TFP (2017=1)', color='tab:orange', linewidth=2.5)
plt.plot(years, meanA_scaled, label='Model mean productivity', color='tab:blue', linewidth=2.5)
plt.title('Aggregate Productivity: Model vs. FRED')
plt.xlabel('Year'); plt.ylabel('Index (2017=1)'); plt.legend(); plt.tight_layout()
plt.show()

# -------- plot trajectories with top/bottom node highlighted ---------
idx_top = A_time[-1].argmax()
idx_bot = A_time[-1].argmin()
plt.figure(figsize=(10,5))
for i in range(n_nodes):
    plt.plot(years, A_time[:,i]*scale_factor, color='gray', alpha=0.25, linewidth=0.5)
plt.plot(years, A_time[:, idx_top]*scale_factor, color='red', linewidth=2, label=f'Top node {idx_top}')
plt.plot(years, A_time[:, idx_bot]*scale_factor, color='blue', linewidth=2, label=f'Bottom node {idx_bot}')
plt.plot(years, meanA_scaled, color='black', linewidth=2.5, label='Model mean')
plt.plot(years, fred_values, color='tab:orange', linewidth=2.0, label='FRED TFP')
plt.title('Node Trajectories with Extremes Highlighted')
plt.xlabel('Year'); plt.ylabel('Index (2017=1)')
plt.legend(); plt.tight_layout()
plt.show()
