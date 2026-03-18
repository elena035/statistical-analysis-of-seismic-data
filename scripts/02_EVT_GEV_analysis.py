import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import scipy.stats as stats
from scipy.stats import genextreme, beta, gumbel_r, linregress
from scipy.optimize import minimize
from scipy.special import gamma
from numba import njit
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# Ignore specific fit warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0. GLOBAL CONFIGURATION
# =============================================================================

# --- PATHS ---
DATA_PATH = '../data/'
ASSETS_PATH = '../assets/'

if not os.path.exists(ASSETS_PATH):
    os.makedirs(ASSETS_PATH)

# --- GRAPHICS ---
FIG_SIZE_SUBFIGURE = (5, 4)
FIG_SIZE_FULL = (8, 5)
FONT_AXIS = 13
FONT_AXIS_SUB = 14
FONT_TICKS = 11
FONT_LEGEND = 'large'

COLORS = {
    'Deep earthquakes': 'purple',
    'Shallow earthquakes': 'orange'
}

# Apply global plotting style for LaTeX fonts
plt.rcParams.update({
    "text.usetex": True,             
    "font.family": "serif",          
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
})

# =============================================================================
# 1. CORE FUNCTIONS
# =============================================================================

def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """Calculates distance in km between a point and an array of points (Haversine)."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2_array)
    dphi = np.radians(lat2_array - lat1)
    dlambda = np.radians(lon2_array - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def gml_fit(data):
    """Generalized Maximum Likelihood (GML) estimator."""
    def log_prior(xi):
        x = xi + 0.5
        if 0 < x < 1: return beta.logpdf(x, 6, 9)
        else: return -1e10
        
    def neg_log_posterior(theta):
        xi, loc, scale = theta
        if scale <= 0: return 1e10
        if xi <= -0.5 or xi >= 0.5: return 1e10
        log_lik_data = np.sum(genextreme.logpdf(data, xi, loc=loc, scale=scale))
        return -(log_lik_data + log_prior(xi))
        
    try: initial_guess = genextreme.fit(data)
    except: initial_guess = [0.1, np.mean(data), np.std(data)]
    result = minimize(neg_log_posterior, initial_guess, method='Nelder-Mead')
    return result.x

def estimate_gev_lmoments_martins(data):
    """
    Estimates GEV parameters (xi, alpha, kappa) using L-Moments (Martins & Stedinger, 2000).
    Note: kappa < 0 implies heavy tail (Frechet) in this parameterization.
    """
    x = np.sort(np.array(data, dtype=float))
    n = len(x)
    if n < 3: return np.nan, np.nan, np.nan
    
    b0 = np.mean(x)
    i = np.arange(1, n + 1)
    b1 = np.sum((i - 1) * x) / (n * (n - 1))
    b2 = np.sum((i - 1) * (i - 2) * x) / (n * (n - 1) * (n - 2))
    
    lam1 = b0
    lam2 = 2 * b1 - b0
    lam3 = 6 * b2 - 6 * b1 + b0
    tau3 = lam3 / lam2
    
    # Polynomial approx for kappa
    c = 2 / (3 + tau3) - np.log(2) / np.log(3)
    k = 7.8590 * c + 2.9554 * c**2
    
    gam = gamma(1 + k)
    alpha = (lam2 * k) / ((1 - 2**(-k)) * gam)
    xi_loc = lam1 - (alpha / k) * (1 - gam)
    
    return xi_loc, alpha, k

@njit
def _core_compute_max(times_ns, mags, window_ns, start_time_ns, end_time_ns):
    """Numba optimized block maxima finder."""
    total_duration = end_time_ns - start_time_ns
    n_blocks = int(np.ceil(total_duration / window_ns))
    max_values = np.full(n_blocks, np.nan)
    
    for i in range(len(times_ns)):
        t = times_ns[i]
        if t < start_time_ns or t >= end_time_ns: continue
        m = mags[i]
        block_idx = int((t - start_time_ns) // window_ns)
        if 0 <= block_idx < n_blocks:
            if np.isnan(max_values[block_idx]) or m > max_values[block_idx]:
                max_values[block_idx] = m
    return max_values

def compute_block_maxima(df, time_col, mag_col, windows_dict, global_start, global_end):
    """Wrapper for calculating block maxima."""
    series_time = df[time_col]
    if not np.issubdtype(series_time.dtype, np.datetime64):
        series_time = pd.to_datetime(series_time)
    
    times_array = series_time.astype(np.int64).values
    mags_array = df[mag_col].values
    start_ns = global_start.value
    end_ns = global_end.value
    
    results = {}
    for name, ns_value in windows_dict.items():
        results[name] = _core_compute_max(times_array, mags_array, ns_value, start_ns, end_ns)
    return results

def get_window_parameters(mag, method='GK74_TABLE'):
    """
    Returns distance (km) and time (days) for declustering 
    interpolating Gardner & Knopoff (1974) table.
    """
    if method == 'GK74_TABLE':
        m_vals = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
        L_vals = np.array([19.5, 22.5, 26, 30, 35, 40, 47, 54, 61, 70, 81, 94.0])
        T_vals = np.array([6.0, 11.5, 22, 42, 83, 155, 290, 510, 790, 915, 960, 985])
        
        dist_km = np.interp(mag, m_vals, L_vals)
        time_days = np.interp(mag, m_vals, T_vals)
        return dist_km, time_days
    return 20.0, 100.0 

def decluster_gardner_knopoff(df, time_col='Time', mag_col='Magnitude', lat_col='Latitude', lon_col='Longitude'):
    """Declustering using Gardner & Knopoff (1974) method."""
    catalog = df.copy()
    catalog[time_col] = pd.to_datetime(catalog[time_col])
    catalog['Mw_calc'] = catalog[mag_col].astype(float)
    
    catalog['is_active'] = True
    catalog['cluster_id'] = -1
    
    sorted_indices = catalog.sort_values(by='Mw_calc', ascending=False).index
    
    print(f"Start Declustering GK74 ({len(catalog)} events)...")
    count = 0
    
    for idx in sorted_indices:
        if not catalog.at[idx, 'is_active']: continue
        
        mw = catalog.at[idx, 'Mw_calc']
        t0 = catalog.at[idx, time_col]
        lat0, lon0 = catalog.at[idx, lat_col], catalog.at[idx, lon_col]
        
        dist_km, tau_days = get_window_parameters(mw, method='GK74_TABLE')
        t_end = t0 + pd.Timedelta(days=tau_days)
        
        mask = (catalog[time_col] > t0) & (catalog[time_col] <= t_end) & (catalog['is_active'])
        potentials = catalog.index[mask]
        
        if len(potentials) > 0:
            dists = haversine_vectorized(lat0, lon0, 
                                         catalog.loc[potentials, lat_col].values, 
                                         catalog.loc[potentials, lon_col].values)
            
            to_remove = potentials[dists <= dist_km]
            if len(to_remove) > 0:
                catalog.loc[to_remove, 'is_active'] = False
                catalog.loc[to_remove, 'cluster_id'] = idx
        
        count += 1
        if count % 2000 == 0: print(f"Processed {count} potential mainshocks...")

    df_main = catalog[catalog['is_active']].copy()
    df_cluster = catalog[~catalog['is_active']].copy()
    df_main.drop(columns=['is_active'], inplace=True)
    
    print(f"Done. Mainshocks: {len(df_main)} ({len(df_main)/len(df):.1%})")
    print(f"Removed (Aftershocks/Foreshocks): {len(df_cluster)}")
    
    return df_main, df_cluster    

def test_stationarity(array, name, threshold, regression='ct', maxlag=None):
    """Augmented Dickey-Fuller (ADF) stationarity test."""
    print(f"--- ADF Test: {name} ---")
    if maxlag is None:
        res = adfuller(array, regression=regression, autolag='AIC')
        print("Mode: Augmented Dickey-Fuller (auto-selection AIC)")
    else:
        res = adfuller(array, maxlag=maxlag, regression=regression, autolag=None)
        print(f"Mode: Dickey-Fuller with fixed lag = {maxlag}")
    
    print(f'ADF Statistic: {res[0]:.4e}')
    print(f'p-value: {res[1]:.4e}')
    
    if res[1] < threshold: print("Result: Stationary")
    else: print("Result: Non-Stationary")
    print("-" * 30)
    
    return res

# =============================================================================
# 2. LOAD DATA AND PREPROCESSING
# =============================================================================
print("\n--- 1. Loading and Declustering ---")
try:
    df_deep_raw = pd.read_csv(os.path.join(DATA_PATH, 'earthquakes_deep.csv'))
    df_shallow_raw = pd.read_csv(os.path.join(DATA_PATH, 'earthquakes_shallow.csv'))
    print("Data successfully loaded.")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Error loading file: {e}. Run the preprocessing script first.")

# Declustering
df_shallow_clean, _ = decluster_gardner_knopoff(df_shallow_raw)
df_deep_clean, _ = decluster_gardner_knopoff(df_deep_raw)

# Global Temporal Setup
global_start = min(pd.to_datetime(df_deep_clean['Time']).min(), pd.to_datetime(df_shallow_clean['Time']).min())
global_end = max(pd.to_datetime(df_deep_clean['Time']).max(), pd.to_datetime(df_shallow_clean['Time']).max())
print(f"Aligned temporal analysis: {global_start.date()} -> {global_end.date()}")

# =============================================================================
# 3. VISUAL ANALYSIS: ACF AND SCATTER PLOTS
# =============================================================================

# --- 3.1 ACF PLOT ---
for label, data in datasets_to_check.items():
    data = data[~np.isnan(data)]
    col = COLORS.get(label, 'black')
    
    plt.figure(figsize=FIG_SIZE_SUBFIGURE)
    ax = plt.gca()
    plot_acf(data, ax=ax, lags=15, alpha=0.05, title=None, color=col)
    
    # Manual Styling
    for item in ax.collections:
        if type(item).__name__ == 'PolyCollection':
            item.set_facecolor('lightgray'); item.set_alpha(0.3); item.set_zorder(0)
        elif type(item).__name__ == 'LineCollection':
            item.set_color(col); item.set_linewidth(1.7); item.set_alpha(1.0); item.set_zorder(10)
        elif type(item).__name__ == 'PathCollection':
            item.set_facecolor(col); item.set_edgecolor(col); item.set_sizes([60]); item.set_zorder(11)
            
    for line in ax.lines:
        line.set_color(col); line.set_linewidth(2); line.set_zorder(2)

    ax.set_xlabel("Lag (temporal windows)", fontsize=FONT_AXIS_SUB)
    ax.set_ylabel("Autocorrelation", fontsize=FONT_AXIS_SUB)
    ax.tick_params(labelsize=FONT_TICKS)
    ax.set_ylim(-0.5, 1.1)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=-1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_PATH, f'ACF_{label.split()[0]}_{DAYS_REF}d.png'), dpi=300)
    plt.show()

# --- 3.2 SCATTER PLOT & TREND ---
for label, data in datasets_to_check.items():
    data = data[~np.isnan(data)]
    col = COLORS.get(label, 'black')
    
    plt.figure(figsize=FIG_SIZE_FULL)
    ax = plt.gca()
    
    df_ref = df_deep_clean if 'Deep' in label else df_shallow_clean
    start_time = pd.to_datetime(df_ref['Time']).min()
    time_axis = [start_time + pd.Timedelta(days=DAYS_REF*i) for i in range(len(data))]
    
    start_year_fractional = start_time.year + start_time.dayofyear / 365.25
    x_years_rel = np.array([(t.year + t.dayofyear/365.25) - start_year_fractional for t in time_axis])
    sigma = np.std(data, ddof=1)
    
    ax.scatter(time_axis, data, color=col, alpha=0.6, s=50, label=fr'Block Maxima, $\sigma = {sigma:.2f}$')
    
    slope_yr, intercept, r_val, p_val, std_err_yr = linregress(x_years_rel, data)
    intercept_err = std_err_yr * np.sqrt(np.mean(x_years_rel**2))

    if slope_yr != 0: exponent = int(np.floor(np.log10(np.abs(slope_yr))))
    else: exponent = 0
        
    power_of_10 = 10**exponent
    slope_scaled = slope_yr / power_of_10
    err_scaled = std_err_yr / power_of_10

    eq = (fr"Fit: $M \approx ({intercept:.2f} \pm {intercept_err:.2f}) + "
          fr"({slope_scaled:.2f} \pm {err_scaled:.2f}) \times 10^{{{exponent}}} \cdot t$")
    
    y_fit = intercept + slope_yr * x_years_rel
    ax.plot(time_axis, y_fit, 'r-', linewidth=2.5, label=eq)
    
    ax.set_xlabel("Time t [Years]", fontsize=FONT_AXIS_SUB)
    ax.set_ylabel("Block Maximum Magnitude", fontsize=FONT_AXIS_SUB)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=FONT_TICKS)
    ax.legend(fontsize=FONT_LEGEND, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylim(2.2, 6.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_PATH, f'scatter_trend_{label.split()[0]}_{DAYS_REF}d.png'), dpi=300)
    plt.show()

# Run Stationarity Test on detrended data
for label, data in datasets_to_check.items():
    data = data[~np.isnan(data)]
    _ = test_stationarity(data, label, 0.05, maxlag=0)

# =============================================================================
# 4. MULTI-WINDOW ANALYSIS (PLATEAU TEST)
# =============================================================================
windows_days = np.arange(125, 180, 5)
n_reshuffles = 100
df_target = df_deep_clean
label_target = "Deep Earthquakes"

start_ns = global_start.value
end_ns = global_end.value
results_by_window = {'T_days': [], 'xi_median': [], 'xi_q16': [], 'xi_q84': []}

print(f"\n--- Initiating Stability Analysis on {label_target}... ---")

for days in windows_days:
    window_ns = int(days * 24 * 3600 * 1e9)
    times_orig = pd.to_datetime(df_target['Time']).astype(np.int64).values
    mags_orig = df_target['Magnitude'].values
    
    xi_bootstrap = []
    np.random.seed(42)
    for _ in range(n_reshuffles):
        rand_times = np.random.uniform(times_orig.min(), times_orig.max(), len(times_orig)).astype(np.int64)
        rand_times.sort()
        
        reshuffled_maxima_full = _core_compute_max(rand_times, mags_orig, window_ns, start_ns, end_ns)
        reshuffled_maxima = reshuffled_maxima_full[~np.isnan(reshuffled_maxima_full)]
        n = len(reshuffled_maxima)
        if n > 20:
            try:
                mu_loc, sigma_scale, xi_shape = estimate_gev_lmoments_martins(reshuffled_maxima)
                if -1.0 < xi_shape < 1.0: 
                    xi_bootstrap.append(xi_shape)
            except: pass

    if len(xi_bootstrap) > 0:
        xi_med = np.median(xi_bootstrap)
        xi_16 = np.percentile(xi_bootstrap, 16)
        xi_84 = np.percentile(xi_bootstrap, 84)
        
        results_by_window['T_days'].append(days)
        results_by_window['xi_median'].append(xi_med)
        results_by_window['xi_q16'].append(xi_16)
        results_by_window['xi_q84'].append(xi_84)
        print(f"T = {days} days | N_real = {n} | N_boot = {len(xi_bootstrap)} | xi = {xi_med:.4f} ({xi_16:.3f}, {xi_84:.3f})")
    else:
        print(f"T = {days} days | No valid fits.")

# Plot Plateau
plt.figure(figsize=(8, 5))
x = results_by_window['T_days']
y = results_by_window['xi_median']
y_err_lower = np.array(y) - np.array(results_by_window['xi_q16'])
y_err_upper = np.array(results_by_window['xi_q84']) - np.array(y)

plot_color = 'orange' if label_target == 'Shallow Earthquakes' else 'purple'
plt.errorbar(x, y, yerr=[y_err_lower, y_err_upper], fmt='-o', capsize=5, 
             label=f'L-Moments Estimate', color=plot_color)

plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label=r'Gumbel ($\xi=0$)')
plt.xlabel('Time window T (days)', fontsize=FONT_AXIS)
plt.ylabel(r'Shape parameter $\xi$', fontsize=FONT_AXIS)
plt.legend(fontsize=FONT_LEGEND)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(ASSETS_PATH, f'stability_gev_xi_{label_target.split()[0]}_{DAYS_REF}d_lmom.png'), dpi=300)
plt.show()

# =============================================================================
# 5. GEV ANALYSIS: FULL LOOP & LATEX EXPORT (WITH MU, SIGMA & RMSE)
# =============================================================================
print("\n--- 6. GEV Analysis (Dual Method Loop) ---")
np.random.seed(42)

methods_to_run = ['LMOM', 'GML']
final_results = {label: {} for label in datasets_to_check.keys()}
gev_results_store = {} 

for METHOD in methods_to_run:
    print(f"\n   >>> Executing Method: {METHOD}")
    n_boot, n_sim = 100, 1000

    for label, data in datasets_to_check.items():
        clean_data = data[~np.isnan(data)]
        n_data = len(clean_data)
        if n_data < 10: continue
        print(f"       [{label}] Processing {n_data} blocks...")
        
        df_curr = df_deep_clean if 'Deep' in label else df_shallow_clean
        t_orig = pd.to_datetime(df_curr['Time']).astype(np.int64).values
        mag_col = 'Mw_calc' if 'Mw_calc' in df_curr.columns else 'Magnitude'
        m_orig = df_curr[mag_col].values
        win_ns = windows[window_ref_key]
        s_ns, e_ns = global_start.value, global_end.value

        # --- A. BOOTSTRAP ---
        c_scipy_list, mu_list, sigma_list, valid_sizes = [], [], [], []

        for _ in range(n_boot):
            rand_t = np.random.uniform(t_orig.min(), t_orig.max(), len(t_orig)).astype(np.int64)
            rand_t.sort()
            maxima_boot = _core_compute_max(rand_t, m_orig, win_ns, s_ns, e_ns)
            maxima_boot = maxima_boot[~np.isnan(maxima_boot)]
            
            if len(maxima_boot) > 10:
                valid_sizes.append(len(maxima_boot))
                try:
                    if METHOD == 'LMOM':
                        mu_est, sigma_est, c_est = estimate_gev_lmoments_martins(maxima_boot)
                    elif METHOD == 'GML':
                        c_est, mu_est, sigma_est = gml_fit(maxima_boot)
                    
                    if -1.0 < c_est < 1.0:
                        c_scipy_list.append(c_est)
                        mu_list.append(mu_est)
                        sigma_list.append(sigma_est)
                except: pass
                
        if not c_scipy_list: continue
            
        c_med, mu_med, sigma_med = np.median(c_scipy_list), np.median(mu_list), np.median(sigma_list)
        avg_N = int(np.median(valid_sizes))
        
        # --- B. MONTE CARLO ---
        c_sims = []
        for _ in range(n_sim):
            try:
                synth = genextreme.rvs(c=c_med, loc=mu_med, scale=sigma_med, size=avg_N)
                if METHOD == 'LMOM': _, _, c_sim_val = estimate_gev_lmoments_martins(synth)
                elif METHOD == 'GML': c_sim_val, _, _ = gml_fit(synth)
                if -1.0 < c_sim_val < 1.0: c_sims.append(c_sim_val)
            except: pass
            
        if len(c_sims) > 0:
            q_low, q_high = np.percentile(c_sims, 5), np.percentile(c_sims, 95)
            rmse = np.sqrt(np.mean((np.array(c_sims) - c_med)**2))
            
            final_results[label][METHOD] = {
                'q_low': q_low, 'median': c_med, 'q_high': q_high,
                'rmse': rmse, 'mu': mu_med, 'sigma': sigma_med 
            }
            
            if METHOD == 'GML' or label not in gev_results_store:
                gev_results_store[label] = {
                    'c': c_med, 'mu': mu_med, 'sigma': sigma_med,
                    'xi_q16': q_low, 'xi_q84': q_high, 'method': METHOD
                }

# Export LaTeX
latex_path = os.path.join(DATA_PATH, 'gev_xi_estimates.tex')
with open(latex_path, 'w') as f:
    f.write(r"% Auto-generated LaTeX table" + "\n\\begin{table}[ht]\n\\centering\n")
    f.write(r"\caption{GEV Parameters ($\mu, \sigma, \xi$). Best Estimation and 90\% Monte Carlo CI for $\xi$.}" + "\n")
    f.write(r"\begin{tabular}{llcccccc}" + "\n\\hline\n")
    f.write(r"\textbf{Dataset} & \textbf{Method} & \textbf{$\mu_{med}$} & \textbf{$\sigma_{med}$} & \textbf{$Q_{5}$} & \textbf{$\xi_{best}$} & \textbf{$Q_{95}$} & \textbf{RMSE} \\" + "\n\\hline\n")
    
    for label, methods_data in final_results.items():
        if not methods_data: continue
        f.write(r"\multirow{2}{*}{\textbf{" + label.split()[0] + r"}}" + "\n")
        
        for method in ['LMOM', 'GML']:
            if method in methods_data:
                res = methods_data[method]
                line = f" & {method} & {res['mu']:.2f} & {res['sigma']:.2f} & {res['q_low']:.4f} & \\textbf{{{res['median']:.4f}}} & {res['q_high']:.4f} & {res['rmse']:.4f} \\\\"
                f.write(line + "\n")
            else: f.write(f" & {method} & - & - & - & - & - & - \\\\\n")
        f.write(r"\hline" + "\n")
    f.write(r"\end{tabular}" + "\n\\label{tab:gev_params_results}\n\\end{table}\n")

# Save checkpoint
np.savez(os.path.join(DATA_PATH, 'gev_results_checkpoint.npz'), final_results=final_results, gev_results_store=gev_results_store)

# =============================================================================
# 6. FINAL REPORTS & PLOTTING (PDF, CDF, RISK, RETURN PERIODS)
# =============================================================================
print("\n--- 7. Generating Final Reports & Plots ---")

def get_dist_label(xi_val, xi_low, xi_high):
    if xi_high < 0: return "Weibull (Bounded)"
    elif xi_low > 0: return "Frechet (Heavy Tail)"
    else: return "Gumbel (Compatible)"

for label, data in datasets_to_check.items():
    if label not in final_results: continue
    data = data[~np.isnan(data)]
    
    res = final_results[label].get('GML', gev_results_store.get(label))
    if not res: continue
    
    c_val, mu_val, sigma_val = res.get('median', res.get('c')), res['mu'], res['sigma']
    method_used = 'GML' if 'GML' in final_results[label] else res.get('method', 'Unknown')
    dist_type = get_dist_label(c_val, res.get('q_low', res.get('xi_q16')), res.get('q_high', res.get('xi_q84')))
    
    print(f"[{label}] Plotting {method_used}... Type: {dist_type}")
    col = COLORS.get(label, 'black')

    # A. PDF PLOT
    plt.figure(figsize=FIG_SIZE_SUBFIGURE)
    ax = plt.gca()
    ax.hist(data, bins=9, density=True, color=col, alpha=0.3, label='Observed maxima')
    x_grid = np.linspace(min(data)-0.5, max(data)+1.5, 300)
    ax.plot(x_grid, genextreme.pdf(x_grid, c=c_val, loc=mu_val, scale=sigma_val), color='red', lw=2.5, label=f'GEV Fit ({method_used})\n$\\xi={c_val:.3f}$')
    loc_g, scale_g = gumbel_r.fit(data)
    ax.plot(x_grid, gumbel_r.pdf(x_grid, loc_g, scale_g), 'k:', lw=1.5, label='Gumbel Ref.')
    ax.set_xlabel("Block Maximum Magnitude", fontsize=FONT_AXIS_SUB)
    ax.set_ylabel("Probability Density Functions \n(PDFs)", fontsize=FONT_AXIS_SUB)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.ylim(0, 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_PATH, f'Final_PDF_{label.split()[0]}.png'), dpi=300)
    plt.show()

    # B. CDF PLOT
    plt.figure(figsize=FIG_SIZE_SUBFIGURE)
    sorted_data, n = np.sort(data), len(data)
    prob_emp = (np.arange(1, n+1) - 0.44)/(n+0.12)
    plt.scatter(sorted_data, prob_emp, marker='o', edgecolors=col, facecolors='none', s=20, label='Empirical CDFs')
    plt.plot(x_grid, genextreme.cdf(x_grid, c=c_val, loc=mu_val, scale=sigma_val), color='red', lw=1.5, label='GEV Model')
    plt.plot(x_grid, gumbel_r.cdf(x_grid, loc_g, scale_g), 'k:', label='Gumbel')
    plt.xlabel("Block Maximum Magnitude", fontsize=FONT_AXIS_SUB)
    plt.ylabel("Cumulative Distribution Functions \n(CDFs)", fontsize=FONT_AXIS_SUB)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_PATH, f'Final_CDF_{label.split()[0]}.png'), dpi=300)
    plt.show()

    # C. RETURN PERIOD PLOT (T vs M)
    block_len_years = DAYS_REF / 365.25
    mag_axis = np.linspace(2.5, 9.0, 200)
    plt.figure(figsize=FIG_SIZE_SUBFIGURE)
    T_emp = block_len_years / (1 - prob_emp)
    plt.scatter(sorted_data, T_emp, facecolors='none', edgecolors=col, s=20, label='Empirical')
    gev_cdf_grid = genextreme.cdf(mag_axis, c=c_val, loc=mu_val, scale=sigma_val)
    with np.errstate(divide='ignore'):
        T_curve = block_len_years / (1 - gev_cdf_grid)
    plt.plot(mag_axis, T_curve, color='red', lw=1.5, label='GEV Model')
    plt.yscale('log')
    plt.xlabel("Block Maximum Magnitude", fontsize=FONT_AXIS)
    plt.ylabel(r"Return Period $\tau$ [Years]", fontsize=FONT_AXIS)
    plt.ylim(0.2, 2000)
    plt.xlim(2.5, 8.5)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_PATH, f'Final_ReturnPeriod_{label.split()[0]}.png'), dpi=300)
    plt.show()

    # D. RISK PROBABILITY PLOT
    plt.figure(figsize=FIG_SIZE_SUBFIGURE)
    horizons, styles = [5, 40, 80], ['-', '--', ':']
    mag_risk = np.linspace(2.5, 8.5, 200)
    cdf_risk = genextreme.cdf(mag_risk, c=c_val, loc=mu_val, scale=sigma_val)
    for i, yrs in enumerate(horizons):
        n_blocks = yrs / block_len_years
        risk_curve = 1 - (cdf_risk ** n_blocks)
        plt.plot(mag_risk, risk_curve, color=col, ls=styles[i], lw=2, label=f'{yrs} years')
    plt.xlabel("Block Maximum Magnitude", fontsize=FONT_AXIS)
    plt.ylabel("Exceedance Probability p", fontsize=FONT_AXIS)
    plt.xlim(3.5, 8.5)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_PATH, f'Final_Risk_{label.split()[0]}.png'), dpi=300)
    plt.show()

# =============================================================================
# 7. Q-Q PLOT FINAL
# =============================================================================
print("\n--- 8. Q-Q Plot Validation ---")
plt.figure(figsize=FIG_SIZE_FULL)
ax = plt.gca()

for label, data in datasets_to_check.items():
    data = np.sort(data[~np.isnan(data)])
    n = len(data)
    col = COLORS.get(label, 'black')
    
    # Empirical Quantiles (Gringorten)
    prob = (np.arange(1, n+1) - 0.44)/(n+0.12)
    x_emp = -np.log(-np.log(prob))
    
    slope, intercept, r_val, _, _ = linregress(x_emp, data)
    ax.scatter(x_emp, data, facecolors='none', edgecolors=col, label=f'{label} ($R^2={r_val**2:.3f}$)')
    ax.plot(x_emp, intercept + slope*x_emp, color=col, ls='--')

ax.set_xlabel("Reduced Gumbel Variate", fontsize=FONT_AXIS)
ax.set_ylabel("Block Maximum Magnitude", fontsize=FONT_AXIS)
ax.legend(fontsize=FONT_LEGEND)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, f'Final_QQ.png'), dpi=300)
plt.show()