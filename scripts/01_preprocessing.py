import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, Polygon
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import linregress
import matplotlib.dates as mdates

# Handle scalebar gracefully if not installed
try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError:
    ScaleBar = None

# =============================================================================
# 0. GLOBAL CONFIGURATION & PARAMETERS
# =============================================================================

# --- PATHS (Relative for GitHub) ---
DATA_PATH = '../data/'
ASSETS_PATH = '../assets/'

if not os.path.exists(ASSETS_PATH):
    os.makedirs(ASSETS_PATH)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# --- SCIENTIFIC PARAMETERS ---
MC_THRESHOLD = 2.5       # Completeness Magnitude
THRESHOLD_DEPTH = 70.0   # Shallow/Deep threshold (km)
MARGIN_MAP_M = 10000     # Map margin in meters

# --- GRAPHIC PARAMETERS (Publication Ready) ---
FIG_SIZE_WIDE = (8, 5)   # Wide plots (Time Series, G-R law)
FIG_SIZE_MAP = (5, 5)    # Square plots (Maps)
DPI = 300

# Parameters for "Split" style mapping (Small vs Large events)
SCALE_FACTOR_SMALL = 0.8   
SCALE_FACTOR_LARGE = 0.08  
SPLIT_MAG = 3.5        

# --- FONTS & STYLES ---
FONT_TITLE = 12
FONT_AXIS = 13
FONT_TICKS = 11
FONT_LEGEND = 'large'

# Apply global plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
})

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def add_scalebar(ax):
    """ Adds a scalebar to the map if the library is installed. """
    if ScaleBar is not None:
        try:
            scalebar = ScaleBar(1, units="m", location="lower right", 
                                box_alpha=0.9, font_properties={'size': 9})
            ax.add_artist(scalebar)
        except Exception as e:
            print(f"ScaleBar error: {e}")

# =============================================================================
# 2. RAW DATA LOADING & BOUNDING BOX
# =============================================================================
print("--- PHASE 1: Loading Raw Data ---")

# Look for all .txt files in the data folder (assuming INGV pipe-separated format)
files = sorted(glob.glob(os.path.join(DATA_PATH, '*.txt')))
if not files:
    raise FileNotFoundError(f"No .txt files found in {DATA_PATH}. Please add raw catalog files.")

df_raw = pd.concat([pd.read_csv(f, sep='|') for f in files], ignore_index=True)
df_raw = df_raw.drop_duplicates(keep='first')
df_raw['Time'] = pd.to_datetime(df_raw['Time'])

# Keep only relevant columns
cols_to_keep = ['Time', 'Latitude', 'Longitude', 'Depth/Km', 'Magnitude', 'MagType']
df_raw = df_raw[cols_to_keep].copy()

# --- Bounding Box Filtering (Central-Southern Italy) ---
min_lon, min_lat = 10.0, 36.5
max_lon, max_lat = 18.0, 43.8

df_raw = df_raw[
    (df_raw['Longitude'] >= min_lon) & 
    (df_raw['Longitude'] <= max_lon) & 
    (df_raw['Latitude'] >= min_lat) & 
    (df_raw['Latitude'] <= max_lat)
]

print(f"Total Events (RAW inside Bounding Box): {len(df_raw)}")

# =============================================================================
# 3. GIS PREPARATION & CONTOUR MAP ON RAW DATA
# =============================================================================
print("--- PHASE 2: Plotting Spatial Density (KDE Contour) ---")

geometry = [Point(xy) for xy in zip(df_raw['Longitude'], df_raw['Latitude'])]
gdf = gpd.GeoDataFrame(df_raw, geometry=geometry, crs='EPSG:4326') 
gdf_wm = gdf.to_crs(epsg=3857)

# Sort by magnitude for proper z-order plotting
gdf_sorted = gdf_wm.sort_values(by='Magnitude', ascending=True)

fig, ax = plt.subplots(figsize=FIG_SIZE_MAP) 

bbox_geom = Polygon([(min_lon, min_lat), (max_lon, min_lat), 
                     (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat)])

gdf_bbox = gpd.GeoDataFrame(index=[0], geometry=[bbox_geom], crs='EPSG:4326')
gdf_bbox_wm = gdf_bbox.to_crs(epsg=3857)

# Plot Bounding Box
gdf_bbox_wm.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.0, linestyle='--', zorder=3)

# Zoom and Margin
bounds = gdf_bbox_wm.total_bounds
margin = 20000 
ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

# Add Basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldShadedRelief, zoom='auto')
ax.set_axis_off()

# Density Contours (KDE)
sns.kdeplot(x=gdf_sorted.geometry.x, y=gdf_sorted.geometry.y, ax=ax,
            levels=12, color='#00008B', linewidths=0.7, alpha=0.6, zorder=3)

add_scalebar(ax)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'map_density_contours.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. GUTENBERG-RICHTER PLOT (RAW DATA)
# =============================================================================
print("--- PHASE 3: Gutenberg-Richter Analysis (Raw Data) ---")

df_shallow_raw = df_raw[df_raw['Depth/Km'] <= THRESHOLD_DEPTH].copy()

magnitudes = df_shallow_raw['Magnitude'].values
magnitudes_sorted = np.sort(magnitudes)
N_tot = len(magnitudes)
cdf_magnitudes = np.arange(N_tot, 0, -1) # Cumulative Number (N >= M)

# Calculate parameters only on the complete catalog part (M >= Mc)
mask_complete = magnitudes_sorted >= MC_THRESHOLD
mag_complete = magnitudes_sorted[mask_complete]
n_complete = len(mag_complete)

# MLE (Aki/Utsu) - Robust method
mean_mag = np.mean(mag_complete)
b_mle = np.log10(np.exp(1)) / (mean_mag - (MC_THRESHOLD - 0.05))
a_mle = np.log10(n_complete) + b_mle * MC_THRESHOLD

# Least Squares (For plotting the theoretical line)
y_log_complete = np.log10(cdf_magnitudes[mask_complete])
slope, intercept, _, _, std_err = linregress(mag_complete, y_log_complete)
b_lsq = -slope
sigma_b = std_err

# Plot G-R Law
plt.figure(figsize=FIG_SIZE_WIDE) 

# Empirical Data
plt.scatter(magnitudes_sorted, cdf_magnitudes, s=20, 
            facecolors='none', alpha=0.6, edgecolors='black', label='Empirical data (Raw)')

# Theoretical Line
x_theory = np.linspace(min(magnitudes), max(magnitudes), 100)
y_theory = 10**(intercept + slope * x_theory)

label_fit = (fr'Gutenberg-Richter Law $N \simeq 10^{{-bM}}$' 
             '\n' 
             fr'$b={b_lsq:.3f} \pm {sigma_b:.3f}$')
plt.plot(x_theory, y_theory, 'r-', linewidth=2, label=label_fit)

# Completeness Line
plt.axvline(x=MC_THRESHOLD, color='green', linestyle='--', 
            label=f'Completeness $M_c={MC_THRESHOLD}$')

plt.xlabel('Magnitude (M)', fontsize=FONT_AXIS)
plt.ylabel(r'Cumulative Number of Events ($N \geq M$)', fontsize=FONT_AXIS)
plt.yscale('log')
plt.grid(True, which='both', linestyle='-', alpha=0.3, zorder=0)
plt.legend(fontsize=FONT_LEGEND)
plt.tick_params(labelsize=FONT_TICKS)
plt.tight_layout()

plt.savefig(os.path.join(ASSETS_PATH, 'plot_Gutenberg_Richter_shallow.png'), dpi=DPI) 
plt.show()

# =============================================================================
# 5. DATA CLEANING AND EXPORT (FOR EVT AND IT)
# =============================================================================
print("--- PHASE 4: Filtering Dataset (M >= Mc) ---")

df_clean = df_raw[df_raw['Magnitude'] >= MC_THRESHOLD].copy()

# Split Shallow / Deep
df_shallow = df_clean[df_clean['Depth/Km'] <= THRESHOLD_DEPTH].copy()
df_deep = df_clean[df_clean['Depth/Km'] > THRESHOLD_DEPTH].copy()

# Save Processed CSVs
df_clean.to_csv(os.path.join(DATA_PATH, 'earthquakes_total.csv'), index=False)
df_shallow.to_csv(os.path.join(DATA_PATH, 'earthquakes_shallow.csv'), index=False)
df_deep.to_csv(os.path.join(DATA_PATH, 'earthquakes_deep.csv'), index=False)

print(f"Events after Mc threshold ({MC_THRESHOLD}): {len(df_clean)}")
print(f" -> Shallow: {len(df_shallow)}")
print(f" -> Deep:    {len(df_deep)}")

# =============================================================================
# 6. TIME SERIES: MAGNITUDE AND CUMULATIVE COUNT
# =============================================================================
print("--- PHASE 5: Time Series Analysis ---")

df_plot_time = df_clean.sort_values(by='Time').copy()
df_plot_time['cumulative'] = range(1, len(df_plot_time) + 1)

fig, ax1 = plt.subplots(figsize=FIG_SIZE_WIDE) 

# Scatter Plot (Magnitude) - Left Axis
ax1.scatter(df_plot_time['Time'], df_plot_time['Magnitude'], 
            marker='o', s=20, facecolors='none', edgecolors='black', 
            alpha=0.9, linewidth=0.6, label='Single Event Magnitude')

ax1.set_xlabel('Time t [Years]', fontsize=FONT_AXIS, labelpad=10)
ax1.set_ylabel('Magnitude ($M$)', fontsize=FONT_AXIS, labelpad=10)
ax1.tick_params(axis='both', labelsize=FONT_TICKS)
ax1.grid(True, linestyle=':', alpha=0.5)

# Cumulative Curve - Right Axis
ax2 = ax1.twinx()
ax2.plot(df_plot_time['Time'], df_plot_time['cumulative'], 
         color='red', linewidth=2, label='Cumulative Number N(t)')

ax2.set_ylabel('Cumulative Number of Events $N(t)$', fontsize=FONT_AXIS, color='red', labelpad=15)
ax2.tick_params(axis='y', labelcolor='red', labelsize=FONT_TICKS)

# Time Axis Formatting
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)

# Combine Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, fontsize=FONT_LEGEND)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'time_series_magnitude_cumulative.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. MAGNITUDE & DEPTH MAPS (CLEAN DATA)
# =============================================================================
print("--- PHASE 6: Plotting Final Maps ---")

# Reproject clean data
geometry_clean = [Point(xy) for xy in zip(df_clean['Longitude'], df_clean['Latitude'])]
gdf_clean = gpd.GeoDataFrame(df_clean, geometry=geometry_clean, crs='EPSG:4326') 
gdf_clean_wm = gdf_clean.to_crs(epsg=3857)

gdf_sorted_clean = gdf_clean_wm.sort_values(by='Magnitude', ascending=True)

# Map limits
bounds_clean = gdf_sorted_clean.total_bounds
xlim = (bounds_clean[0] - MARGIN_MAP_M, bounds_clean[2] + MARGIN_MAP_M)
ylim = (bounds_clean[1] - MARGIN_MAP_M, bounds_clean[3] + MARGIN_MAP_M)

# --- MAGNITUDE MAP ---
fig, ax = plt.subplots(figsize=FIG_SIZE_MAP) 

mask_small = gdf_sorted_clean['Magnitude'] < SPLIT_MAG
gdf_small = gdf_sorted_clean[mask_small]
gdf_large = gdf_sorted_clean[~mask_small]

# Plot Small Events (Background)
gdf_small.plot(ax=ax, column='Magnitude', cmap='inferno_r',
               markersize=SCALE_FACTOR_SMALL, alpha=0.9, edgecolor='none',
               vmin=MC_THRESHOLD, vmax=6.0, zorder=1)

# Plot Large Events (Foreground with edge)
marker_sizes_large = (gdf_large['Magnitude']**3) * SCALE_FACTOR_LARGE
gdf_large.plot(ax=ax, column='Magnitude', cmap='inferno_r',
               markersize=marker_sizes_large, alpha=0.9, 
               edgecolor='black', linewidth=0.1,
               vmin=MC_THRESHOLD, vmax=6.0, zorder=2)

ctx.add_basemap(ax, source=ctx.providers.Esri.WorldShadedRelief, zoom=8)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_axis_off()
add_scalebar(ax)

cbar = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=MC_THRESHOLD, vmax=6.0), cmap='inferno_r'),
                    ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
cbar.set_label('Magnitude (M)', fontsize=14)
cbar.ax.tick_params(labelsize=FONT_TICKS)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'map_magnitude_distribution.png'), dpi=DPI, bbox_inches='tight')
plt.show()

# --- DEPTH MAP ---
fig, ax = plt.subplots(figsize=FIG_SIZE_MAP) 

gdf_depth_sorted = gdf_clean_wm.sort_values(by='Depth/Km', ascending=True)

mask_small_depth = gdf_depth_sorted['Magnitude'] < SPLIT_MAG
gdf_small_depth = gdf_depth_sorted[mask_small_depth]
gdf_large_depth = gdf_depth_sorted[~mask_small_depth]

CMAP_DEPTH = 'viridis_r'
VMAX_DEPTH = 100

gdf_small_depth.plot(ax=ax, column='Depth/Km', cmap=CMAP_DEPTH,
                     markersize=SCALE_FACTOR_SMALL, alpha=0.9, edgecolor='none',
                     vmin=0, vmax=VMAX_DEPTH, zorder=1)

marker_sizes_large_depth = (gdf_large_depth['Magnitude']**3) * SCALE_FACTOR_LARGE
gdf_large_depth.plot(ax=ax, column='Depth/Km', cmap=CMAP_DEPTH,
                     markersize=marker_sizes_large_depth, alpha=0.9, 
                     edgecolor='black', linewidth=0.1,
                     vmin=0, vmax=VMAX_DEPTH, zorder=2)

ctx.add_basemap(ax, source=ctx.providers.Esri.WorldShadedRelief, zoom=8)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_axis_off()
add_scalebar(ax)

cbar = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=VMAX_DEPTH), cmap=CMAP_DEPTH),
                    ax=ax, orientation='vertical', fraction=0.03, pad=0.02, extend='max')
cbar.set_label('Focus Depth [km]', fontsize=14)
cbar.ax.tick_params(labelsize=FONT_TICKS)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_PATH, 'map_depth_distribution.png'), dpi=DPI, bbox_inches='tight')
plt.show()