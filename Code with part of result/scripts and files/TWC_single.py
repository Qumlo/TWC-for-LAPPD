#!/usr/bin/env python
# coding: utf-8


import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import warnings
import os
import sys
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def predict_fit_value(model, tot_value):
    return model.predict([[tot_value]])[0]

def import_data(file_path, hist_name):
    root_file = uproot.open(file_path)
    hist = root_file[hist_name]
    w, x_edges, y_edges = hist.to_numpy()
    return w, x_edges, y_edges 

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

def fit_gaussian(ToA_diff, ToA_hist, initial_params):
    try:
        popt, pcov = curve_fit(gaussian, ToA_diff, ToA_hist, p0=initial_params, maxfev=10000000)
        sigma = popt[2]
        error = np.sqrt(np.diag(pcov))[2]
    except (RuntimeError, ValueError) as e:
        sigma = np.nan
        error = np.nan
    return sigma, error, popt if not np.isnan(sigma) else None


def process_channel(file_path, hist_name_template, channel):
    hist_name = hist_name_template.format(channel)
    w, ToT_edges, ToA_edges = import_data(file_path, hist_name)

    y_min = -130
    y_max = -100
    y_indices = np.where((ToA_edges >= y_min) & (ToA_edges <= y_max))[0]
    if ToA_edges[y_indices[-1]] > y_max:
        y_indices = y_indices[:-1]

    w_sliced = w[:, y_indices[:-1]]
    ToA_edges_sliced = ToA_edges[y_indices]
    ToA_diff = (ToA_edges_sliced[:-1] + ToA_edges_sliced[1:]) / 2
    ToA_hist = np.sum(w_sliced, axis=0)

    initial_params = [ToA_hist.max(), ToA_diff[np.argmax(ToA_hist)], 1]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            popt, pcov = curve_fit(gaussian, ToA_diff, ToA_hist, p0=initial_params, maxfev=10000000)
            initial_sigma = popt[2]
            initial_error = np.sqrt(np.diag(pcov))[2]
        except (RuntimeError, ValueError) as e:
            initial_sigma = np.nan
            initial_error = np.nan

        if len(w) > 0:
            for warning in w:
                pass

    num_bins = 40
    ToT_bin_edges = np.linspace(ToT_edges[0], ToT_edges[-1], num_bins + 1)
    ToT_bin_centers = (ToT_bin_edges[:-1] + ToT_bin_edges[1:]) / 2

    # Step 1: Find the ToT bin with the most events
    total_events_per_bin = []
    for i in range(num_bins):
        bin_mask = (ToT_edges[:-1] >= ToT_bin_edges[i]) & (ToT_edges[:-1] < ToT_bin_edges[i + 1])
        bin_weights = w_sliced[bin_mask, :].sum(axis=0)
        total_events_per_bin.append(np.sum(bin_weights))

    max_bin_index = np.argmax(total_events_per_bin)
    max_bin_center = ToT_bin_centers[max_bin_index]

    # Step 2: Select data around the max bin center ±1.1
    fit_range_min = max_bin_center - 1.1
    fit_range_max = max_bin_center + 1.1

    toa_diff_means = []
    toa_diff_errors = []
    ToT_valid_centers = []

    for i in range(num_bins):
        bin_min = ToT_bin_edges[i]
        bin_max = ToT_bin_edges[i + 1]
        if bin_min < fit_range_min or bin_max > fit_range_max:
            continue  # Skip bins outside the range

        bin_mask = (ToT_edges[:-1] >= bin_min) & (ToT_edges[:-1] < bin_max)
        if np.any(bin_mask):
            bin_weights = w_sliced[bin_mask, :].sum(axis=0)
        else:
            bin_weights = np.zeros_like(ToA_edges_sliced[:-1])
        if bin_weights.sum() > 0:
            bin_centers = (ToA_edges_sliced[:-1] + ToA_edges_sliced[1:]) / 2
            try:
                popt_gauss, _ = curve_fit(gaussian, bin_centers, bin_weights,
                                          p0=[np.max(bin_weights), bin_centers[np.argmax(bin_weights)], 1],
                                          maxfev=1000000)
                mean = popt_gauss[1]
                N = bin_weights.sum()
                error = np.sqrt(N) / N
                ToT_valid_centers.append(ToT_bin_centers[i])
                toa_diff_means.append(mean)
                toa_diff_errors.append(error)
            except (RuntimeError, ValueError) as e:
                pass  # Suppress individual bin fit warnings

    ToT_valid_centers = np.array(ToT_valid_centers)
    toa_diff_means = np.array(toa_diff_means)
    toa_diff_errors = np.array(toa_diff_errors)

    if len(ToT_valid_centers) > 0:
        degree = 3
        alpha = 10

        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
        model.fit(ToT_valid_centers[:, np.newaxis], toa_diff_means, ridge__sample_weight=1 / toa_diff_errors ** 2)

        def predict_fit_value(tot_value):
            return model.predict([[tot_value]])[0]

        corrected_ToA_values = []
        for i in range(len(ToT_edges) - 1):
            for j in range(len(ToA_edges_sliced) - 1):
                if w_sliced[i, j] > 0:
                    ToT_value = (ToT_edges[i] + ToT_edges[i + 1]) / 2
                    if fit_range_min <= ToT_value <= fit_range_max:
                        ToA_value = (ToA_edges_sliced[j] + ToA_edges_sliced[j + 1]) / 2
                        corrected_ToA_value = ToA_value - predict_fit_value(ToT_value)
                        corrected_ToA_values.append(corrected_ToA_value)

        corrected_ToA_values = np.array(corrected_ToA_values)
        corrected_ToA_min = corrected_ToA_values.min()
        corrected_ToA_max = corrected_ToA_values.max()
        ToA_edges_corrected = np.linspace(corrected_ToA_min, corrected_ToA_max, len(ToA_edges_sliced))

        corrected_w = np.zeros_like(w_sliced)
        for i in range(len(ToT_edges) - 1):
            for j in range(len(ToA_edges_sliced) - 1):
                if w_sliced[i, j] > 0:
                    ToT_value = (ToT_edges[i] + ToT_edges[i + 1]) / 2
                    if fit_range_min <= ToT_value <= fit_range_max:
                        ToA_value = (ToA_edges_sliced[j] + ToA_edges_sliced[j + 1]) / 2
                        corrected_ToA_value = ToA_value - predict_fit_value(ToT_value)
                        if corrected_ToA_min <= corrected_ToA_value <= corrected_ToA_max:
                            corrected_j = np.digitize(corrected_ToA_value, ToA_edges_corrected) - 1
                            corrected_j = min(max(corrected_j, 0), len(ToA_edges_corrected) - 2)
                            corrected_w[i, corrected_j] += w_sliced[i, j]

        ToA_diff_corrected = (ToA_edges_corrected[:-1] + ToA_edges_corrected[1:]) / 2
        ToA_hist_corrected = np.sum(corrected_w, axis=0)

        initial_params_corrected = [ToA_hist_corrected.max(), ToA_diff_corrected[np.argmax(ToA_hist_corrected)], 1]
        try:
            popt_corrected, pcov_corrected = curve_fit(gaussian, ToA_diff_corrected, ToA_hist_corrected,
                                                       p0=initial_params_corrected, maxfev=1000000)
            corrected_sigma = popt_corrected[2]
            corrected_error = np.sqrt(np.diag(pcov_corrected))[2]
        except (RuntimeError, ValueError) as e:
            corrected_sigma = np.nan
            corrected_error = np.nan
    else:
        corrected_sigma = np.nan
        corrected_error = np.nan

    return initial_sigma, corrected_sigma, initial_error, corrected_error, w_sliced, ToT_edges, ToA_edges_sliced, corrected_w, ToA_edges_corrected, model, fit_range_max, fit_range_min


# In[3]:


def process_channel_ToT(file_path, hist_name_template, channel):
    hist_name = hist_name_template.format(channel)
    y_min = -130
    y_max = -100
    w, ToT_edges, ToA_edges = import_data(file_path, hist_name)

    y_indices = np.where((ToA_edges >= y_min) & (ToA_edges <= y_max))[0]
    if ToA_edges[y_indices[-1]] > y_max:
        y_indices = y_indices[:-1]

    w_sliced = w[:, y_indices[:-1]]
    ToA_edges_sliced = ToA_edges[y_indices]

    ToT_hist = np.sum(w_sliced, axis=1)
    ToT_bin_centers = (ToT_edges[:-1] + ToT_edges[1:]) / 2

    # weighted mean
    weighted_mean_ToT = np.sum(ToT_bin_centers * ToT_hist) / np.sum(ToT_hist)

    # median
    cumulative_hist = np.cumsum(ToT_hist)
    median_index = np.searchsorted(cumulative_hist, cumulative_hist[-1] / 2)
    median_ToT = ToT_bin_centers[median_index]

    # mode
    mode_index = np.argmax(ToT_hist)
    mode_ToT = ToT_bin_centers[mode_index]
    
    # mean

    if len(ToT_hist) > 0 and len(ToT_bin_centers) > 0:
        popt_ToT, _ = curve_fit(gaussian, ToT_bin_centers, ToT_hist, p0=[ToT_hist.max(), ToT_bin_centers[np.argmax(ToT_hist)], 1], maxfev=1000000)
        mean_ToT = popt_ToT[1]
    else:
        mean_ToT = np.nan

    return mean_ToT, weighted_mean_ToT, median_ToT, mode_ToT

def analyze_file_ToT(file_path):
    mean_ToT_values = []
    weighted_mean_ToT_values = []
    median_ToT_values = []
    mode_ToT_values = []

    for channel in range(64):
        if channel == 1 or channel == 7 or channel == 21:
            continue
        
        mean_ToT, weighted_mean_ToT, median_ToT, mode_ToT = process_channel_ToT(file_path, 'ToA_ToT_intime_ch{0};1', channel)
        mean_ToT_values.append(mean_ToT)
        weighted_mean_ToT_values.append(weighted_mean_ToT)
        median_ToT_values.append(median_ToT)
        mode_ToT_values.append(mode_ToT)
    
    return mean_ToT_values, weighted_mean_ToT_values, median_ToT_values, mode_ToT_values


# In[6]:


def process_channel_ToA(file_path, hist_name_template, channel):
    hist_name = hist_name_template.format(channel)
    w, ToT_edges, ToA_edges = import_data(file_path, hist_name)

    ToA_hist = np.sum(w, axis=0)
    ToA_bin_centers = (ToA_edges[:-1] + ToA_edges[1:]) / 2

    weighted_mean_ToA = np.sum(ToA_bin_centers * ToA_hist) / np.sum(ToA_hist)

    cumulative_hist = np.cumsum(ToA_hist)
    median_index = np.searchsorted(cumulative_hist, cumulative_hist[-1] / 2)
    median_ToA = ToA_bin_centers[median_index]

    mode_index = np.argmax(ToA_hist)
    mode_ToA = ToA_bin_centers[mode_index]

    if len(ToA_hist) > 0 and len(ToA_bin_centers) > 0:
        popt_ToA, pcov = curve_fit(gaussian, ToA_bin_centers, ToA_hist, p0=[ToA_hist.max(), ToA_bin_centers[np.argmax(ToA_hist)], 1], maxfev=1000000)
        mean_ToA = popt_ToA[1]
        sigma_ToA = popt_ToA[2]
        sigma_error = np.sqrt(np.diag(pcov))[2]
    else:
        mean_ToA = np.nan
        sigma_ToA = np.nan
        sigma_error = np.nan

    return mean_ToA, weighted_mean_ToA, median_ToA, mode_ToA

def analyze_file_ToA(file_path):
    mean_ToA_values = []
    weighted_mean_ToA_values = []
    median_ToA_values = []
    mode_ToA_values = []

    for channel in range(64):
        if channel == 1 or channel == 7 or channel == 21:
            continue
        
        mean_ToA, weighted_mean_ToA, median_ToA, mode_ToA = process_channel_ToA(file_path, 'ToA_ToT_intime_ch{0};1', channel)
        mean_ToA_values.append(mean_ToA)
        weighted_mean_ToA_values.append(weighted_mean_ToA)
        median_ToA_values.append(median_ToA)
        mode_ToA_values.append(mode_ToA)
    
    return mean_ToA_values, weighted_mean_ToA_values, median_ToA_values, mode_ToA_values
#mean_ToA_0412_avg = np.nanmean(mean_ToA_values_0412)
#mean_ToA_0419_avg = np.nanmean(mean_ToA_values_0419)
#mean_ToA_0427_avg = np.nanmean(mean_ToA_values_0427)

#print("0412 Gaussian Mean ToA Average:", mean_ToA_0412_avg)
#print("0419 Gaussian Mean ToA Average:", mean_ToA_0419_avg)
#print("0427 Gaussian Mean ToA Average:", mean_ToA_0427_avg)


# In[ ]:


def main(file_path):
    output_dir = "../TWC_Result"
    os.makedirs(output_dir, exist_ok=True)
    initial_sigma_values = []
    corrected_sigma_values = []
    initial_sigma_values_intime = []
    corrected_sigma_values_intime = []
    initial_errors = []
    corrected_errors = []
    initial_errors_intime = []
    corrected_errors_intime = []
    hist_name_template = 'ToA_ToT_intime_ch{0};1'
    pdf_path = os.path.join(output_dir, 'Fitting.pdf')

    output_combined_file = os.path.join(output_dir, 'combined_histograms.root')

    mean_ToT_values, weighted_mean_ToT_values, median_ToT_values, mode_ToT_values = analyze_file_ToT(file_path)
    mean_ToA_values, weighted_mean_ToA_values, median_ToA_values, mode_ToA_values = analyze_file_ToA(file_path)

    with uproot.recreate(output_combined_file) as f:
        with PdfPages(pdf_path) as pdf:
            for channel in range(64):
                if channel == 1 or channel == 7 or channel == 21:
                    continue

                initial_sigma, corrected_sigma, initial_error, corrected_error, w_sliced, ToT_edges, ToA_edges_sliced, corrected_w, ToA_edges_corrected, model, fit_range_max, fit_range_min = process_channel(file_path, 'ToA_ToT_intime_ch{0};1', channel)
                initial_sigma_values.append(initial_sigma)
                corrected_sigma_values.append(corrected_sigma)
                initial_errors.append(initial_error)
                corrected_errors.append(corrected_error)

                initial_sigma_intime, corrected_sigma_intime, initial_error_intime, corrected_error_intime, _, _, _, _, _, _, _, _ = process_channel(file_path, 'ToA_ToT_ch{0};1', channel)
                initial_sigma_values_intime.append(initial_sigma_intime)
                corrected_sigma_values_intime.append(corrected_sigma_intime)
                initial_errors_intime.append(initial_error_intime)
                corrected_errors_intime.append(corrected_error_intime)

                ToT_mask = (ToT_edges[:-1] >= fit_range_min) & (ToT_edges[:-1] <= fit_range_max)
                ToT_edges_limited = ToT_edges[:-1][ToT_mask]
                w_sliced_limited = w_sliced[ToT_mask, :]

                cmap = plt.cm.viridis
                cmap.set_under(color='white')

                plt.figure(figsize=(8, 6))
                plt.pcolormesh(ToT_edges, ToA_edges_sliced, w_sliced.T, shading='auto', cmap=plt.cm.viridis, vmin=1)
                plt.colorbar(label='Counts')
                plt.xlabel('ToT [ns]')
                plt.ylabel('ToA difference [ns]')
                plt.title(f'2D Histogram with Fitting curve (Channel {channel})')
                fit_ToA_centers = (ToA_edges_sliced[:-1] + ToA_edges_sliced[1:]) / 2
                fit_curve_limited = [predict_fit_value(model, t) for t in ToT_edges_limited]
                plt.plot(ToT_edges_limited, fit_curve_limited, 'r-', label='Fit Curve')
                plt.legend()
                pdf.savefig()
                plt.close()


                # Plot Gaussian fit of ToA Difference
                ToA_diff = (ToA_edges_sliced[:-1] + ToA_edges_sliced[1:]) / 2
                ToA_hist = np.sum(w_sliced, axis=0)
                initial_params = [ToA_hist.max(), ToA_diff[np.argmax(ToA_hist)], 1]
                popt, pcov = curve_fit(gaussian, ToA_diff, ToA_hist, p0=initial_params, maxfev=1000000)
                x_smooth = np.linspace(ToA_diff.min(), ToA_diff.max(), 1000)

                plt.figure(figsize=(10, 6))
                plt.plot(ToA_diff, ToA_hist, 'b-', label='Data')
                plt.plot(x_smooth, gaussian(x_smooth, *popt), 'r--', label=f'Gaussian fit (σ = {popt[2]:.4f} ns)')
                plt.xlabel('ToA difference [ns]')
                plt.ylabel('Counts')
                plt.title(f'Gaussian Fit of ToA Difference (Channel {channel})')
                plt.legend()
                plt.grid(True)
                pdf.savefig()  # Save the current figure into the PDF
                plt.close()

                # Plot Gaussian fit of Corrected ToA Difference
                ToA_diff_corrected = (ToA_edges_corrected[:-1] + ToA_edges_corrected[1:]) / 2
                ToA_hist_corrected = np.sum(corrected_w, axis=0)
                initial_params_corrected = [ToA_hist_corrected.max(), ToA_diff_corrected[np.argmax(ToA_hist_corrected)], 1]
                popt_corrected, pcov_corrected = curve_fit(gaussian, ToA_diff_corrected, ToA_hist_corrected, p0=initial_params_corrected, maxfev=1000000)
                x_smooth_corrected = np.linspace(ToA_diff_corrected.min(), ToA_diff_corrected.max(), 1000)

                plt.figure(figsize=(10, 6))
                plt.plot(ToA_diff_corrected, ToA_hist_corrected, 'b-', label='Corrected Data')
                plt.plot(x_smooth_corrected, gaussian(x_smooth_corrected, *popt_corrected), 'r--', label=f'Gaussian fit (Corrected) (σ = {popt_corrected[2]:.4f} ns)')
                plt.xlabel('Corrected ToA difference [ns]')
                plt.ylabel('Counts')
                plt.title(f'Gaussian Fit of Corrected ToA Difference (Channel {channel})')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()

                # Save histograms to ROOT file
                f[f"original_hist_ch{channel}"] = (w_sliced, ToT_edges, ToA_edges_sliced)
                f[f"corrected_hist_ch{channel}"] = (corrected_w, ToT_edges, ToA_edges_corrected)

                f[f"gaussian_fit_ch{channel}"] = (ToA_hist, ToA_edges_sliced)
                f[f"gaussian_fit_corrected_ch{channel}"] = (ToA_hist_corrected, ToA_edges_corrected)

        channels = np.arange(64)
        channels = np.delete(channels, [1, 7, 21])

        f["mean_ToT_values"] = {"channels": channels, "mean_ToT_values": mean_ToT_values}
        f["mean_ToA_values"] = {"channels": channels, "mean_ToA_values": mean_ToA_values}



    channels = np.arange(64)
    channels = np.delete(channels, [1, 7, 21])
    plt.figure(figsize=(12, 8))
    #plt.errorbar(channels, initial_sigma_values, yerr=initial_errors, fmt='o', color='blue', label='Initial Sigma (ToA_ToT)', markersize=3)
    plt.errorbar(channels, initial_sigma_values_intime, yerr=initial_errors_intime, fmt='o', color='green', label='Initial Sigma (ToA_ToT_intime)', markersize=3)
    #plt.errorbar(channels, corrected_sigma_values, yerr=corrected_errors, fmt='o', color='red', label='Corrected Sigma (ToA_ToT)', markersize=3)
    plt.errorbar(channels, corrected_sigma_values_intime, yerr=corrected_errors_intime, fmt='o', color='purple', label='Corrected Sigma (ToA_ToT_intime)', markersize=3)

    #plt.plot(channels, initial_sigma_values, 'b--')
    plt.plot(channels, initial_sigma_values_intime, 'g--')
    #plt.plot(channels, corrected_sigma_values, 'r--')
    plt.plot(channels, corrected_sigma_values_intime, 'm--')

    plt.xlabel('Channel')
    plt.ylabel('Sigma [ns]')
    plt.title('Initial and Corrected Sigma Values for Each Channel')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'sigma_combined_with_error_bars.png'))

    # ToT and ToA
    mean_ToT_values, weighted_mean_ToT_values, median_ToT_values, mode_ToT_values = analyze_file_ToT(file_path)
    mean_ToA_values, weighted_mean_ToA_values, median_ToA_values, mode_ToA_values = analyze_file_ToA(file_path)


    plt.figure(figsize=(12, 8))
    plt.plot(channels, mean_ToT_values, 'o-', label='Gaussian Mean ToT')
    plt.xlabel('Channel')
    plt.ylabel('Mean ToT [ns]')
    plt.title('Mean ToT Values for Each Channel')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mean_tot_values.png'))

    plt.figure(figsize=(12, 8))
    plt.plot(channels, mean_ToA_values, 'o-', label='Gaussian Mean ToA')
    plt.xlabel('Channel')
    plt.ylabel('Mean ToA [ns]')
    plt.title('Mean ToA Values for Each Channel')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mean_ToA_values.png'))



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 TWC_single.py /path/to/rootfile")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = "../TWC_Result"
    main(file_path)

