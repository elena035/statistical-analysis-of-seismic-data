import numpy as np
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.dates as mdates
import datetime
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError:
    ScaleBar = None

# =============================================================================
# 0. GLOBAL CONFIGURATION (PUBLICATION READY)
# =============================================================================

# --- PATHS ---
DATA_PATH = '../data/'
ASSETS_PATH = '../assets/'
INPUT_FILE = os.path.join(DATA_PATH, 'sciame_sismico_amatrice4.txt')

if not os.path.exists(ASSETS_PATH):
    os.makedirs(ASSETS_PATH)

# --- GRAPHIC CONFIGURATION ---
DPI = 300
FIG_SIZE_WIDE = (8, 5)    # For Time Series
FIG_SIZE_MAP = (4, 5)     # For Maps
FIG_SIZE_FS = (7, 6)      # For Fisher-Shannon Plane

FONT_TITLE = 12
FONT_AXIS = 13
FONT_TICKS = 11
FONT_LEGEND = 'large'
FONT_INSET = 8

# Apply global plotting style for LaTeX fonts
plt.rcParams.update({
    "text.usetex": True,             
    "font.family": "serif",          
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
})

# --- MAINSHOCKS (Central Italy 2016-2017 Swarm) ---
MAINSHOCKS = {
    'Amatrice (M 6.0)': '2016-08-24 01:36:32',
    'Visso (M 5.9)': '2016-10-26 19:18:05',
    'Norcia (M 6.5)': '2016-10-30 06:40:18',
    'Campotosto (M 5.5)': '2017-01-18 10:14:09'
}

EVENT_COLORS = ['red', 'green', 'm', 'blue']

# --- STATISTICAL ANALYSIS PARAMETERS ---
WINDOW_SIZE = 100   # Moving window size (number of events)
STEP_SIZE = 2       # Window step
BINS_SHANNON = 5    # Bins for Shannon/Fisher
BINS_NMI = 5        # Bins for NMI

COLOR_SHANNON = 'black'
COLOR_FISHER = 'black'
COLOR_NMI = 'black'

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def add_scalebar(ax):
    """ Adds a scalebar to the map. """
    if ScaleBar is not None:
        try:
            scalebar = ScaleBar(1, units="m", location="lower right", 
                                box_alpha=0.9, font_properties={'size': 8},
                                bbox_transform=ax.transAxes)
            ax.add_artist(scalebar)
        except Exception as e:
            print(f"ScaleBar error: {e}")

# =============================================================================
# 2. DATA LOADING AND PREPARATION
# =============================================================================
print("Loading data...")
df_master = pd.read_csv(INPUT_FILE, sep='|')

# Clean unused columns
cols_to_drop = ['Catalog', 'Contributor', 'ContributorID', 'MagAuthor', 'Author', 'EventType']
df_master = df_master.drop(columns=[c for c in cols_to_drop if c in df_master.columns])

# Temporal Conversion and Sorting
df_master['Time'] = pd.to_datetime(df_master['Time'])
df_master = df_master.sort_values('Time').reset_index(drop=True)

# Calculate inter-event times (Delta_t)
df_master['delta_t'] = df_master['Time'].diff().dt.total_seconds()

# Clean data for statistical analysis
df_work = df_master.dropna(subset=['delta_t']).copy()
df_work = df_work[df_work['delta_t'] > 0]

print(f"Dataset loaded: {len(df_master)} total events.")
print(f"Dataset for statistical analysis (delta_t > 0): {len(df_work)} events.")

# =============================================================================
# 3. GEOGRAPHIC MAP (Geopandas + Contextily)
# =============================================================================
print("Generating Swarm Map...")

mc_threshold = df_master['Magnitude'].min()

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(df_master['Longitude'], df_master['Latitude'])]
gdf = gpd.GeoDataFrame(df_master, geometry=geometry, crs='EPSG:4326')
gdf_wm = gdf.to_crs(epsg=3857)

# Filter and sort
gdf_plot = gdf_wm[gdf_wm['Magnitude'] >= mc_threshold].sort_values(by='Magnitude', ascending=True)

# Marker size based on magnitude
marker_sizes = (gdf_plot['Magnitude']**3) * 0.3

# --- Bounding Box ---
min_lon, min_lat = 12.90, 42.30
max_lon, max_lat = 13.60, 43.10
bbox_geom = Polygon([(min_lon, min_lat), (max_lon, min_lat), 
                     (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat)])

gdf_bbox = gpd.GeoDataFrame(index=[0], geometry=[bbox_geom], crs='EPSG:4326')
gdf_bbox_wm = gdf_bbox.to_crs(epsg=3857)

# --- Plotting ---
fig, ax = plt.subplots(figsize=FIG_SIZE_MAP)

# Earthquake Points
gdf_plot.plot(ax=ax, column='Magnitude', cmap='inferno_r', markersize=marker_sizes,
              alpha=0.8, edgecolor='none', vmin=mc_threshold, vmax=6.5, zorder=2)

# Bounding Box
gdf_bbox_wm.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.0, linestyle='--', zorder=3)

# Zoom and Basemap
bounds = gdf_bbox_wm.total_bounds
margin = 20000 
ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=15)
ax.set_axis_off()

add_scalebar(ax)

# --- Inset Map (Italy) ---
ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower left", borderpad=1.5)
try:
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    italy = world[world['NAME'] == "Italy"]
    if italy.empty: italy = world[world['ADMIN'] == "Italy"]
    
    italy.plot(ax=ax_inset, facecolor='lightgray', edgecolor='white')
    gdf_bbox.plot(ax=ax_inset, color='red', alpha=1) 
except Exception as e:
    ax_inset.text(0.5, 0.5, "Map Error", ha='center')

ax_inset.set_title("Location", fontsize=FONT_LEGEND)
ax_inset.set_xticks([]); ax_inset.set_yticks([])
ax_inset.set_facecolor('white'); ax_inset.patch.set_alpha(1)
ax_inset.set_xlim(6, 19); ax_inset.set_ylim(36, 47.5)

# --- Colorbar ---
ax_cbar = inset_axes(ax, width="4%", height="70%", loc='center right', 
                     bbox_transform=ax.transAxes, borderpad=0)
norm = mcolors.Normalize(vmin=mc_threshold, vmax=6.5)
cbar = fig.colorbar(cm.ScalarMappable(cmap='inferno_r', norm=norm), cax=ax_cbar)
cbar.set_label('Magnitude ($M$)', fontsize=FONT_AXIS)
cbar.ax.tick_params(labelsize=FONT_TICKS)

plt.savefig(os.path.join(ASSETS_PATH, 'map_amatrice_swarm.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. TIME SERIES: MAGNITUDE AND CUMULATIVE
# =============================================================================
print("Generating Time Series...")

df_plot_time = df_master.copy()
df_plot_time['cumulative'] = range(1, len(df_plot_time) + 1)

fig, ax1 = plt.subplots(figsize=FIG_SIZE_WIDE)

# Scatter Plot (Magnitude)
ax1.scatter(df_plot_time['Time'], df_plot_time['Magnitude'], 
            marker='o', s=20, facecolors='none', edgecolors='black', 
            alpha=0.6, linewidth=0.6, label='Event')

ax1.set_xlabel('Date (Year-Month)', fontsize=FONT_AXIS, labelpad=5)
ax1.set_ylabel('Magnitude', fontsize=FONT_AXIS, labelpad=5)
ax1.tick_params(axis='both', labelsize=FONT_TICKS)
ax1.grid(True, linestyle=':', alpha=0.5)

# Cumulative Curve
ax2 = ax1.twinx()
ax2.plot(df_plot_time['Time'], df_plot_time['cumulative'], 
         color='red', linewidth=2.0, label='Cumulative Number N(t)')

ax2.set_ylabel('Cumulative Number $N(t)$', fontsize=FONT_AXIS, color='red', labelpad=10)
ax2.tick_params(axis='y', labelcolor='red', labelsize=FONT_TICKS)

# Mainshocks Vertical Lines
for i, (name, date_str) in enumerate(MAINSHOCKS.items()):
    d = pd.to_datetime(date_str)
    if df_plot_time['Time'].min() <= d <= df_plot_time['Time'].max():
        ax1.axvline(x=d, color=EVENT_COLORS[i], linestyle='--', alpha=0.7, linewidth=1, label=f'{name}')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fig.autofmt_xdate(rotation=30)
ax1.set_xlim(datetime.datetime(2016, 1, 1), datetime.datetime(2018, 1, 1))

ax1.legend(loc='upper left', frameon=True, framealpha=0.6, edgecolor='gray', 
           fontsize=FONT_LEGEND, shadow=False)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'swarm_time_series_cumulative.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. STATISTICAL CALCULATIONS
# =============================================================================

def calculate_shannon_entropy(data, bins, normalize=True):
    counts, _ = np.histogram(data, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_max = np.log(len(p))
    if normalize:
        H_max = np.log(bins)
        return H / H_max
    return H, H_max

def calculate_fisher_information(data, bins):
    """
    Calculates Discrete Fisher Information.
    The factor 4 derives from amplitude probability formulation (q = sqrt(p)).
    """
    counts, _ = np.histogram(data, bins=bins)
    p_full = counts / counts.sum() 
    return 4 * np.sum((np.sqrt(p_full[1:]) - np.sqrt(p_full[:-1]))**2)

def calculate_normalized_mi(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    p_xy = c_xy / np.sum(c_xy) + 1e-10
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    H_x = -np.sum(p_x * np.log(p_x))
    H_y = -np.sum(p_y * np.log(p_y))
    H_xy = -np.sum(p_xy * np.log(p_xy))
    MI = H_x + H_y - H_xy
    denom = (H_x + H_y)
    if denom == 0: return 0
    return (2 * MI) / denom

# =============================================================================
# 6. ANALYSIS 1: SHANNON ENTROPY
# =============================================================================
print("Calculating Shannon Entropy...")

entropy_results, time_results, H_massimi = [], [], []
delta_ts = df_work['delta_t'].values
times = df_work['Time'].values

for i in range(0, len(df_work) - WINDOW_SIZE, STEP_SIZE):
    window_data = delta_ts[i : i + WINDOW_SIZE]
    h_val, H_max = calculate_shannon_entropy(window_data, BINS_SHANNON, normalize=False)
    entropy_results.append(h_val)
    H_massimi.append(H_max)
    time_results.append(times[i + WINDOW_SIZE])

fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
ax.plot(time_results, entropy_results, color=COLOR_SHANNON, linewidth=2, label='Norm. Shannon Entropy ($H_n$)')

ax.set_ylabel(r'Normalized Shannon Entropy ($H_{\delta}$)', fontsize=16)
ax.set_xlabel('Date (year-month)', fontsize=16)

ax.tick_params(axis='both', labelsize=FONT_TICKS)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=1)

# --- INSET ZOOM ---
zoom_start = pd.to_datetime('2017-01-10')
zoom_end = pd.to_datetime('2017-01-31')

axins = inset_axes(ax, width="40%", height="30%", loc='lower right',
                   bbox_to_anchor=(0.05, 0.04, 0.91, 1), bbox_transform=ax.transAxes)

axins.plot(time_results, entropy_results, color=COLOR_SHANNON, linewidth=1.5)
axins.set_xlim(zoom_start, zoom_end)
axins.set_ylim(0.0, 0.8) 

axins.set_title("Zoom: Campotosto (18 Jan 2017)", fontsize=10)
axins.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
axins.xaxis.set_major_locator(mdates.DayLocator(interval=4))
axins.tick_params(labelsize=FONT_INSET)
axins.grid(True, linestyle=':', alpha=0.7)
mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.4", ls='--', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'shannon_entropy_normalized.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. ANALYSIS 2: FISHER INFORMATION
# =============================================================================
print("Calculating Fisher Information...")

fisher_results = []
for i in range(0, len(df_work) - WINDOW_SIZE, STEP_SIZE):
    window_data = delta_ts[i : i + WINDOW_SIZE]
    f_val = calculate_fisher_information(window_data, BINS_SHANNON)
    fisher_results.append(f_val)

fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
ax.plot(time_results, fisher_results, color=COLOR_FISHER, linewidth=2, label='Fisher Information ($I$)')

ax.set_ylabel(r'Fisher Information ($I_{\delta}$)', fontsize=16)
ax.set_xlabel('Date (year-month)', fontsize=16)
ax.tick_params(labelsize=FONT_TICKS)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=1)

# --- INSET ZOOM ---
axins = inset_axes(ax, width="40%", height="40%", loc='upper right', 
                   bbox_to_anchor=(0, -0.08, 1, 1), bbox_transform=ax.transAxes)

axins.plot(time_results, fisher_results, color=COLOR_FISHER, linewidth=1.5)
axins.set_xlim(zoom_start, zoom_end)
axins.set_ylim(0.2, 4.15)

axins.set_title("Zoom: Campotosto \n(18 Jan 2017)", fontsize=10)
axins.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
axins.xaxis.set_major_locator(mdates.DayLocator(interval=4))
axins.tick_params(labelsize=FONT_INSET)
axins.grid(True, linestyle=':', alpha=0.5)
mark_inset(ax, axins, loc1=3, loc2=2, fc="none", ec="0.4", ls='--', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'fisher_info_time.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. ANALYSIS 3: FISHER-SHANNON POWER PLANE
# =============================================================================
print("Generating Fisher-Shannon Power Plane...")

entropy_results = np.array(entropy_results)
fisher_results = np.array(fisher_results)
time_results = np.array(time_results)

# Shannon Power (Nx)
h_raw_series = entropy_results * np.log(BINS_SHANNON)
nx_results = np.exp(2 * h_raw_series)

fig, ax = plt.subplots(figsize=FIG_SIZE_FS)

cbar_min_date = pd.to_datetime('2016-06-01')
cbar_max_date = pd.to_datetime('2017-06-01')
cbar_vmin = mdates.date2num(cbar_min_date)
cbar_vmax = mdates.date2num(cbar_max_date)
time_nums = mdates.date2num(time_results)

# Cramér-Rao Bound (I * Nx >= 1)
x_theo = np.logspace(np.log10(min(nx_results)*0.5), np.log10(max(nx_results)*1.5), 200)
y_theo = 1.0 / x_theo

ax.plot(x_theo, y_theo, color='black', linestyle='--', linewidth=2, alpha=0.8, 
        label='Cramér-Rao Bound ($I_{{\delta}} \cdot N_{{\delta}} = 1$)')
ax.fill_between(x_theo, y_theo, 1e-10, color='gray', alpha=0.3, zorder=0)

# Continuous Trajectory via LineCollection
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

SMOOTH_WINDOW = 5
nx_smooth = moving_average(nx_results, SMOOTH_WINDOW)
fi_smooth = moving_average(fisher_results, SMOOTH_WINDOW)
time_smooth = time_nums[SMOOTH_WINDOW-1:] 

points = np.array([nx_smooth, fi_smooth]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = Normalize(vmin=cbar_vmin, vmax=cbar_vmax)
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(time_smooth) 
lc.set_linewidth(1.2)    
lc.set_alpha(0.9)      
lc.set_label('Mean seismic trajectory')
line = ax.add_collection(lc)

ax.set_xlim(min(nx_smooth)*0.5, max(nx_smooth)*2.0)
ax.set_ylim(min(fi_smooth)*0.5, max(fi_smooth)*3.0)

cbar = plt.colorbar(line, ax=ax, extend='both', fraction=0.04, pad=0.04)

# Start & End Points
ax.scatter(nx_results[0], fisher_results[0], s=50, facecolors='black', edgecolors='black', 
           marker='o', linewidth=1, zorder=150, label='Start point')
ax.scatter(nx_results[-1], fisher_results[-1], s=50, facecolors='black', edgecolors='black', 
           marker='X', linewidth=1, zorder=150, label='End point')

# Mainshocks mapping
markers = ['D', 'P', '^', 's'] 
for i, (name, date_str) in enumerate(MAINSHOCKS.items()):
    target_date = pd.to_datetime(date_str)
    start_idx = np.abs(pd.to_datetime(time_results) - target_date).argmin()
    
    search_range = 50 
    end_idx = min(start_idx + search_range, len(fisher_results))
    fisher_slice = fisher_results[start_idx : end_idx]
    
    if len(fisher_slice) > 0: best_idx = start_idx + np.argmax(fisher_slice)
    else: best_idx = start_idx
        
    nx_val = nx_results[best_idx]
    i_val = fisher_results[best_idx]
    
    current_zorder = 100 - i
    ax.scatter(nx_val, i_val, s=120, color=EVENT_COLORS[i], marker=markers[i], 
               edgecolors='black', linewidth=1.5, zorder=current_zorder, alpha=1.0, label=name)

# Formatting
cbar.ax.set_ylabel('Date (year-month)', labelpad=10, fontsize=FONT_AXIS)
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax.set_xlabel(r'Shannon Entropy Power ($N_{{\delta}} = e^{2H_{{\delta}}}$)', fontsize=16)
ax.set_ylabel(r'Fisher Information ($I_{{\delta}}$)', fontsize=16)

ax.set_xscale('log') 
ax.set_yscale('log')

ax.tick_params(axis='both', labelsize=FONT_TICKS, which='major')
ax.grid(True, linestyle=':', alpha=0.6, which='both')
ax.legend(loc='lower left', frameon=True, fontsize='medium', markerscale=0.7)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'fisher_shannon_power_plane.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. ANALYSIS 4: NORMALIZED MUTUAL INFORMATION
# =============================================================================
print("Calculating Normalized Mutual Information (Memory)...")

nmi_results, time_mi = [], []
dt_current = delta_ts[:-1]
dt_next = delta_ts[1:]
times_aligned_mi = times[1:] 

for i in range(0, len(dt_current) - WINDOW_SIZE, STEP_SIZE):
    w_curr = dt_current[i : i + WINDOW_SIZE]
    w_next = dt_next[i : i + WINDOW_SIZE]
    val = calculate_normalized_mi(w_curr, w_next, bins=BINS_NMI)
    nmi_results.append(val)
    time_mi.append(times_aligned_mi[i + WINDOW_SIZE])

fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
ax.plot(time_mi, nmi_results, color=COLOR_NMI, linewidth=1.5, label='Normalized Memory (NMI)')

for i, (name, date_str) in enumerate(MAINSHOCKS.items()):
    d = pd.to_datetime(date_str)
    if d <= pd.to_datetime(time_mi[-1]):
        ax.axvline(x=d, color=EVENT_COLORS[i], linestyle='--', alpha=0.8, linewidth=1.5, label=name)

ax.set_ylabel('Normalized Mutual Information', fontsize=FONT_AXIS)
ax.set_xlabel('Date (year-month)', fontsize=FONT_AXIS)
ax.set_ylim(0, 0.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30, ha='right', fontsize=FONT_TICKS)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=FONT_LEGEND)
ax.set_xlim(pd.to_datetime('2016-08-01'), pd.to_datetime('2017-08-01'))

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'mutual_info.png'), dpi=DPI, bbox_inches='tight')
plt.show()

print("\n--- ANALYSIS COMPLETED ---")