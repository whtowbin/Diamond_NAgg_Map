#%%
import argparse
import matplotlib.pyplot as plt
import pandas as pd
# import os
from pathlib import Path
import numpy as np
from datetime import datetime
import pybaselines as pybl
import xarray as xr
import imageio
import seaborn as sns
from scipy.optimize import lsq_linear

from ftir_map import LoadOmnicMAP as omnic

# Reference files (edit as needed)
parent_dir = Path(__file__).parent
TYPEIIA_PATH = parent_dir / "Quiddit_Spectra_Files" / "typeIIa.csv"
if not TYPEIIA_PATH.exists():
    raise FileNotFoundError(f"TypeIIa reference file not found at {TYPEIIA_PATH}")
CAXBD_PATH = parent_dir / "Quiddit_Spectra_Files" / "CAXBD.csv"
if not CAXBD_PATH.exists():
    raise FileNotFoundError(f"CAXBD reference file not found at {CAXBD_PATH}")  

# TYPEIIA_PATH = "Quiddit_Spectra_Files/typeIIa.csv"
# CAXBD_PATH = "Quiddit_Spectra_Files/CAXBD.csv"

def parse_args():
    parser = argparse.ArgumentParser(description="Process FTIR .map files for Nitrogen aggregation fits.")
    parser.add_argument("map_path", type=str, help="Path to the .map file")
    parser.add_argument("--output", type=str, default="Results", help="Output folder for results")
    parser.add_argument("--lam", type=float, default=2e7, help="Baseline fit parameter lam (default: 2e7)")
    parser.add_argument("--p", type=float, default=0.000001, help="Baseline fit parameter p (default: 0.000001)")
    parser.add_argument("--n_examples", type=int, default=5, help="Number of example spectra to plot (default: 5)")
    return parser.parse_args()

def micron_to_dpi(microns):
    inches = (microns/ 1000000 * 39.37)
    return 1/inches

def baseline2(spectrum, lam=2e7, p=0.000001):
    baseline = pybl.whittaker.asls(spectrum, lam=lam, p=p)[0]
    return baseline

def interpolate_to_common_grid(da):
    wn_low = np.round(da.wn[0], decimals=0)
    wn_high = np.round(da.wn[-1], decimals=0)
    wn_new = np.arange(wn_low, wn_high, 1)
    da_interp = da.interp(wn=wn_new, method="cubic")
    return da_interp

def mask_for_fit(Spectrum):
    masked = Spectrum.where(
        ((Spectrum.wn > 1800) & (Spectrum.wn < 2313))
        | (Spectrum.wn > 2390) & (Spectrum.wn < 2670)
    )
    return masked

def fit_CAXBD(spectrum, CAXBD_matrix):
    bounds = np.array(
        [
            (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf),
            (-np.inf, np.inf), (-np.inf, np.inf),
        ]
    ).T
    try:
        params = lsq_linear(CAXBD_matrix, spectrum, bounds=bounds)["x"]
    except ValueError as e:
        print(e)
        print(spectrum.shape)
        print("Value Error")
        params = np.zeros(5)
    return xr.DataArray(params)

def plot_n_agg_fit(xi, yi, normalized_spectra, N_Fit_Map_DS, CAXBD_Spectra_np, wn_array, output_folder, filename, date_time, ratio):
    """
    Plot the nitrogen aggregation fit for a single spectrum at (xi, yi),
    including individual component fits.
    """
    # Extract measured spectrum
    spec = normalized_spectra.isel(x=xi, y=yi).sel(wn=wn_array)
    # Extract fit parameters
    fit_params = (N_Fit_Map_DS/ratio).isel(x=xi, y=yi)
    fit_params_array = fit_params.to_array().values

    # Individual component fits (C, A, X, B, D)
    component_labels = ["C", "A", "X", "B", "D"]
    component_colors = ["purple", "green", "orange", "brown", "gray"]
    component_fits = [CAXBD_Spectra_np[:, i] * fit_params_array[i] for i in range(5)]

    # Reconstruct total fit spectrum
    fit_spectrum = np.dot(CAXBD_Spectra_np, fit_params_array)

    # Plot
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(wn_array, spec.values, label="Measured (Baseline Subtracted)", color="blue")
    ax.plot(wn_array, fit_spectrum, label="CAXBD Total Fit", color="red", linestyle="--")
    for i, comp_fit in enumerate(component_fits):
        ax.plot(wn_array, comp_fit, label=f"{component_labels[i]} Component", color=component_colors[i], linestyle=":", linewidth = 3)
    ax.set_title(f"N Aggregation Fit (x={xi}, y={yi})")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Absorbance")
    ax.set_xlim(800, 1430)
    ax.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{filename}_N-Agg_fit_example_{xi}_{yi}_{date_time}.png")
    plt.close()


def process_map(
    map_path,
    output_folder,
    typeiia_path=TYPEIIA_PATH,
    caxbd_path=CAXBD_PATH,
    baseline_lam=2e7,
    baseline_p=0.000001,
    n_examples=5
):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    filename = Path(map_path).stem
    now = datetime.now()
    date_time = now.strftime("d-%m-%d-%Y_t-%H-%M")

    # Plot settings
    font = {"weight": "normal", "size": "18"}
    plt.rc("font", **font)
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Load map
    map = omnic.Load_Omnic_Map(map_path)
    stacked = map.stack(allpoints=("x", "y"))
    mid_x_IDX = round(len(map.x)/2)
    mid_y_IDX = round(len(map.y)/2)

    # Interpolate to common grid
    interpolated = interpolate_to_common_grid(stacked).unstack("allpoints")
    # Use baseline_lam and baseline_p from arguments
    def baseline2_param(spectrum):
        return baseline2(spectrum, lam=baseline_lam, p=baseline_p)
    baselines = xr.apply_ufunc(
        baseline2_param, map.spectra, input_core_dims=[["wn"]], output_core_dims=[["wn"]], vectorize=True
    )
    baselines_interp = interpolate_to_common_grid(baselines)
    baseline_subtracted = interpolated - baselines_interp

    # --- Save baseline fit plots for a few example spectra ---
    example_coords = [
        (mid_x_IDX-10, mid_y_IDX-10),
        (mid_x_IDX-10, mid_y_IDX+10),
        (mid_x_IDX+10, mid_y_IDX-10),
        (mid_x_IDX+10, mid_y_IDX+10),
        (mid_x_IDX, mid_y_IDX)
    ]
    for i, (xi, yi) in enumerate(example_coords[:n_examples]):
        fig, ax = plt.subplots()
        map.spectra.isel(x=xi, y=yi).plot(ax=ax, label="Original")
        baselines.isel(x=xi, y=yi).plot(ax=ax, label="Baseline")
        # (map.spectra.isel(x=xi, y=yi) - baselines.isel(x=xi, y=yi)).plot(ax=ax, label="Baseline Subtracted")
        ax.set_title(f"Baseline Fit Example (x={xi}, y={yi})")
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Absorbance")
        ax.legend()
        ax.set_xlim(600, 4000)
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{filename}_baseline_fit_example_{i+1}_{date_time}.png")
        plt.close()

    # Reference spectra
    typeIIA = pd.read_csv(typeiia_path, names=["wn", "absorbance"]).set_index("wn").to_xarray()
    typeIIA_interp = interpolate_to_common_grid(typeIIA)

    # Mask for fit
    Ideal_masked = mask_for_fit(typeIIA_interp).absorbance
    spec_masked = mask_for_fit(baseline_subtracted)
    ratio = (spec_masked.spectra / Ideal_masked).mean("wn")
    stdev = (spec_masked.spectra).std("wn") / spec_masked.spectra.mean("wn")
    normalized_spectra = (baseline_subtracted.spectra / ratio)

    for i, (xi, yi) in enumerate(example_coords):  # TODO Split into two plots with baseline subtracted and typeIIa reference # 
        fig, ax = plt.subplots()
        # Baseline-subtracted spectrum at this point
        # spec = baseline_subtracted.spectra.isel(x=xi, y=yi)
        spec = normalized_spectra.isel(x=xi, y=yi)
        spec.plot(ax=ax, label="Baseline Subtracted")
        # TypeIIa reference, interpolated and masked to same wn range
        typeIIa_ref = typeIIA_interp.absorbance.sel(wn=spec.wn)
        typeIIa_ref.plot(ax=ax, label="TypeIIa Reference")
        ax.set_title(f"TypeIIa vs Baseline Subtracted (x={xi}, y={yi})")
        ax.set_xlim(1400, 4000)
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Absorbance")
        ax.legend()
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{filename}_typeIIa_fit_example_{i+1}_{date_time}.png")
        plt.close() 
    
    # Filter spectra (optional, can be commented out)
    filtered_normalized_spectra =  normalized_spectra.where(
        (baseline_subtracted.spectra.mean("wn") > 0.05)
        & (baseline_subtracted.spectra.mean("wn") < 0.1) 
        & (stdev < 1)
    )

    # Load CAXBD reference
    CAXBD_Spectra = pd.read_csv(caxbd_path, names=["wn", "C", "A", "X", "B", "D"]).set_index("wn")
    wn_low = float(baseline_subtracted.wn[0].values)
    wn_high = 1400
    CAXBD_Spectra = CAXBD_Spectra[wn_low:wn_high]
    CAXBD_Spectra_np = CAXBD_Spectra.to_numpy()
    wn_array = CAXBD_Spectra.index.to_numpy()
    offset = np.ones_like(wn_array)
    linear = np.arange(len(offset)) - (wn_high - wn_low)
    linear_array = np.vstack((offset, linear))
    CAXBD_Spectra_np = np.hstack((CAXBD_Spectra_np, linear_array.T))

    # Fit all spectra in the map
    N_Fit_Map = xr.apply_ufunc(
        lambda s: fit_CAXBD(s, CAXBD_Spectra_np),
        baseline_subtracted.loc[{"wn": slice(wn_low, wn_high)}].spectra,
        input_core_dims=[["wn"]],
        output_core_dims=[["params"]],
        vectorize=True,
    )

    N_Fit_Map_DS = N_Fit_Map.to_dataset("params")
    N_Fit_Map_DS = N_Fit_Map_DS.rename_vars({0: "C", 1: "A", 2: "X", 3: "B", 4: "D"})


    for xi, yi in example_coords[:n_examples]:
        plot_n_agg_fit(
            xi, yi,
            normalized_spectra,
            N_Fit_Map_DS,
            CAXBD_Spectra_np,
            wn_array,
            output_folder,
            filename,
            date_time,
            ratio
        )
        
    # Calculate ppm and save images
    A_Center_ppm = N_Fit_Map_DS.A * 16.5 / ratio
    B_Center_ppm = N_Fit_Map_DS.B * 79.4 / ratio
    B_percent = B_Center_ppm / (B_Center_ppm + A_Center_ppm) * 100
    Total_N = A_Center_ppm + B_Center_ppm


    # # TODO plot spectral fits for example spectra

    # for i, (xi, yi) in enumerate(example_coords):  # 
    #     fig, axs = plt.subplots()
    #     # Baseline-subtracted spectrum at this point
    #     # spec = baseline_subtracted.spectra.isel(x=xi, y=yi)
    #     spec = normalized_spectra.isel(x=xi, y=yi)
    #     spec.plot(ax=ax, label="Baseline Subtracted")

    #     N_Fit_Comp = N_Fit_Map_DS.isel(x=xi, y=yi)
    #     return N_Fit_Comp

        
    #     # TypeIIa reference, interpolated and masked to same wn range
    #     # typeIIa_ref = typeIIA_interp.absorbance.sel(wn=spec.wn)
    #     # typeIIa_ref.plot(ax=ax, label="TypeIIa Reference")

    #     # ax.set_title(f"TypeIIa vs Baseline Subtracted (x={xi}, y={yi})")
    #     ax.set_xlim(600, 1400)
    #     ax.set_xlabel("Wavenumber")
    #     ax.set_ylabel("Absorbance")
    #     ax.legend()
    #     plt.gca().invert_xaxis()
    #     plt.tight_layout()
    #     plt.savefig(f"{output_folder}/{filename}_N-Agg_fit_example_{i+1}_{date_time}.png")
    #     plt.close() 


    sns_cmap = sns.color_palette("mako", as_cmap=True)
    for arr, name in [
        (A_Center_ppm, "A_Center_ppm_Map"),
        (B_Center_ppm, "B_Center_ppm_Map"),
        (B_percent, "B_percent_Map"),
        (N_Fit_Map_DS.C, "C_Map"),
        (N_Fit_Map_DS.D, "D_Map"),
        (Total_N, "Total_N_Map")
    ]:
        ax = arr.plot(cmap=sns_cmap if "Map" in name else None)
        ax.axes.set_aspect("equal")
        plt.savefig(f"{output_folder}/{filename}_{name}_{date_time}.png")
        imageio.imwrite(f"{output_folder}/{filename}_{name}_{date_time}.tif", arr.to_numpy().astype('float16'))
        plt.close()

    print(f"All maps saved to {output_folder}")
#%%
if __name__ == "__main__":
    args = parse_args()
    process_map(args.map_path, args.output)
#%%
# comp = process_map(
#     "/Users/henrytowbin/Projects/A-Center Diamond  N3 Lifetime/CBP-0261/CBP-0261_Map_50umApt_25umStep_4wnRes_8scans_4-1-24_.map",
#     "Results",
#     baseline_lam=1e7,   # adjust as needed
#     baseline_p=0.00001, # adjust as needed
#     n_examples=3        # number of baseline fit plots to save
# )
# %%
