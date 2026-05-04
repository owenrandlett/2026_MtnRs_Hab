# %% Import libraries and setup fodlders:
import os
import sys
import pickle
import glob
import warnings

import numpy as np
import gspread
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import natsort
from itertools import combinations

from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d
import scipy.stats as stats
import scikit_posthocs as sp

# Update system paths for local modules if needed
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.path.dirname(os.path.abspath(""))
sys.path.append(current_dir)
sys.path.append(
    os.path.realpath(os.path.join(current_dir, "ExtraFunctions", "glasbey-master"))
)
from glasbey import Glasbey
import HabTrackFunctions
import csv
import fnmatch
import importlib


# Set up color palette
gb = Glasbey()

# directory of datasets

root_dir = os.path.realpath(r"Q:\2026_MtnrManuscript\BigRigData")

# assume all subdirectories are one level down and start with 2025*

exp_dirs = glob.glob(os.path.realpath(root_dir + "\\202*"))
data_analysis_suffix = "_response_data.pkl"

# % Parameters for analysis
n_blocks = 4
n_stim_blocks = 60
CurvatureStdThresh = 1.7  # max std in curvature trace that is acceptable. non-tracked fish have noisy traces
SpeedStdThresh = 3.5
AngVelThresh = (
    np.pi / 2.5
)  # the max angular velocity per frame we will accept as not an artifact

OBendThresh = 3
# radians of curvature to call an O-bend
CBendThresh = 1
# radians of curvature to call an C-bend

sav_ord = 3  # parameters for the sgolayfilt of curvature
sav_sz = 15
SpeedSmooth = 5
# the kernel of the medfilt1 filter for the speed

camera_rez = 180 / 2290  # mm/pixel

n_init = 3  # what is used as the 'naive' response to the first n_init stmuli

graph_dir = os.path.join(root_dir, "graphs_HabAnalysis")

if not os.path.exists(graph_dir):
    os.mkdir(graph_dir)

#% % Scan through raw data files and extract responses, etc. Does not need to be run if data has already been analyzed and saved. This will only work if the computer has access to the google sheet with the ROIs via gspread
# parameters:
big_rig3 = False # set to true if analyzing data from big rig 3, which has a different google sheet with ROIs than big rig 2. If false, will use big rig 2 sheet. If you have access to both sheets, you can analyze both sets of data and combine them in the end, since the same ROIs are used across sheets for the same experiments.
re_analyze = False  # set to true to re-analyze raw data files.

if re_analyze:
    for exp_dir in tqdm(exp_dirs):

        os.chdir(exp_dir)
        with open("exp_data.pkl", "rb") as f:
            exp_data = pickle.load(f)

        n_fish = np.max(exp_data["im_rois"])

        gc = gspread.oauth()
        if big_rig3:
            sh_str = "1s4Ga1y04dhXehxpoK9JZk7MeFrTOy8_vwkNz021Yw7Y"
        else:
            sh_str = "1YbSu9YZSB-gUrskkQn57ANPkuZJFkafUa9SHnCkkCz4"
        sh = gc.open_by_key(sh_str)
        worksheet = sh.get_worksheet(0)
        df = pd.DataFrame(worksheet.get_all_records())

        path = os.path.normpath(exp_dir)
        ExpDate = path.split(os.sep)[-1][:8]

        for plate in range(2):
            rows = df.loc[
                (df["Date Screened"] == int(ExpDate)) & (df["Plate (0 or 1)"] == plate)
            ]

            # %
            n_groups = rows.shape[0]
            # %
            if n_groups == 0:
                print("\a")
                warnings.warn(
                    "didnt find any entries for "
                    + ExpDate
                    + ", plate number: "
                    + str(plate)
                )
                continue

            # note that ROIs are 1 indexed in the spreadsheet

            rois = []
            names = []
            plates = []

            for i in range(n_groups):
                roi_str = rows["ROIs"].iloc[i]
                if not roi_str == "[]" and not roi_str == "":  # make sure it isnt empty
                    names.append(rows["Group Name"].iloc[i])
                    rois.append(HabTrackFunctions.convert_roi_str(roi_str))
                    plates.append(plate)
            # % get burst trials:

            trials = natsort.natsorted(
                glob.glob(
                    os.path.join(exp_dir, "_plate_" + str(plate) + "*BurstTracks.pkl")
                )
            )
            n_trials = len(trials)

            if n_trials == 0:
                print("\a\n\a\n\a\n\a\n\a\n\a\n\a\n\a")
                print(
                    "didnt find any track files for "
                    + ExpDate
                    + ", plate number: "
                    + str(plate)
                )
                print("... skipping")
                continue

            # %
            stim_given = []
            stim_frame = []
            for i in range(n_trials):
                ind = trials[i].find("stim_type") + 10
                stim_str = trials[i][ind : ind + 2]
                stim_given.append(stim_str)
            stim_given = np.array(stim_given)
            stim_given[stim_given == "df"] = 1  # training dark flashes == 1
            stim_given[stim_given == "om"] = 0
            stim_given[stim_given == "tp"] = 2  # taps = 2

            stim_given[-n_stim_blocks:] = 3  # re-test block = 3
            stim_given = stim_given.astype(int)

            track_data = {
                "Probability_Of_Response": np.zeros((n_trials, n_fish)),
                "Latency_(msec)": np.zeros((n_trials, n_fish)),
                "Proportion_of_Double_Responses": np.zeros((n_trials, n_fish)),
                "Reorientation_(deg)": np.zeros((n_trials, n_fish)),
                "Displacement_(mm)": np.zeros((n_trials, n_fish)),
                "Movement_Duration_(msec)": np.zeros((n_trials, n_fish)),
                "Bend_Amplitude_(deg)": np.zeros((n_trials, n_fish)),
                "Proportion_of_Compound_Responses": np.zeros((n_trials, n_fish)),
                "C1_Bend_Duration_(msec)": np.zeros((n_trials, n_fish)),
                "C1_Angular_Velocity_(deg/msec)": np.zeros((n_trials, n_fish)),
                "TiffFrameInds": [],
                "names": names,
                "plates": plates,
                "rois": rois,
                "spreadsheet": rows,
                "stim_given": stim_given,
            }

            # %
            for trial in tqdm(range(n_trials)):

                trial_file = trials[trial]
                burst_frame = int(
                    trial_file[
                        trial_file.find("burst_frame_") + 12 : trial_file.find("_time_")
                    ]
                )
                track_data["TiffFrameInds"].append(burst_frame)
                tail_coords, orientations, heading_dir, bend_amps = (
                    HabTrackFunctions.load_burst_pkl(trial_file)
                )
                frame_rate = bend_amps.shape[1]  # number of fps

                # % fish are considered not to be tracked properly if they are
                # not found in more than 5% of frames in the movie,
                # or if the curvature or speed trace is too noisy

                delta_orient_trace = np.vstack(
                    (np.full(n_fish, np.nan), np.diff(orientations.T, axis=0))
                )

                delta_orient_trace[delta_orient_trace > np.pi] = (
                    2 * np.pi - delta_orient_trace[delta_orient_trace > np.pi]
                )
                delta_orient_trace[delta_orient_trace < -np.pi] = (
                    delta_orient_trace[delta_orient_trace < -np.pi] + 2 * np.pi
                )
                delta_orient_trace[abs(delta_orient_trace) > AngVelThresh] = np.nan
                curve = bend_amps.T
                curve_smooth = savgol_filter(
                    HabTrackFunctions.ffill_cols(curve), sav_sz, sav_ord, axis=0
                )

                # calculate speed

                x_coors = tail_coords[0, :, 0, :].T
                y_coors = tail_coords[1, :, 1, :].T

                # diff_x = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(x_coors), sav_sz, sav_ord, axis=0), axis=0)
                # diff_y = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(y_coors), sav_sz, sav_ord, axis=0), axis=0)

                diff_x = np.diff(
                    median_filter(HabTrackFunctions.ffill_cols(x_coors), size=(11, 1)),
                    axis=0,
                )
                diff_y = np.diff(
                    median_filter(HabTrackFunctions.ffill_cols(y_coors), size=(11, 1)),
                    axis=0,
                )

                speed = savgol_filter(
                    np.sqrt(np.square(diff_x) + np.square(diff_x)), sav_sz, sav_ord, axis=0
                )
                speed = np.vstack((np.zeros(n_fish), speed))

                obend_start = np.full([n_fish], np.nan)
                obend_happened = np.full([n_fish], np.nan)
                obend_dorient = np.full([n_fish], np.nan)
                obend_disp = np.full([n_fish], np.nan)
                obend_dur = np.full([n_fish], np.nan)
                obend_max_curve = np.full([n_fish], np.nan)
                obend_second_counter = np.full([n_fish], np.nan)
                obend_multibend = np.full([n_fish], np.nan)
                obend_ang_vel = np.full([n_fish], np.nan)
                obend_c1len = np.full([n_fish], np.nan)

                # fish_not_tracked = (np.mean(np.isnan(tail_coords[0,:,0,:]), axis=1) > 0.10) | (np.nanstd(bend_amps, axis=1) > CurvatureStdThresh) | (np.nanstd(speed, axis=0) > 3)
                fish_not_tracked = (
                    np.mean(np.isnan(tail_coords[0, :, 0, :]), axis=1) > 0.5
                )  # | (np.nanstd(bend_amps, axis=1) > CurvatureStdThresh) | (np.nanstd(speed, axis=0) > SpeedStdThresh)

                for fish in range(n_fish):
                    peakind_curve_pos = find_peaks(curve_smooth[:, fish], width=5)[0]
                    peak_curve_pos = curve_smooth[peakind_curve_pos, fish]

                    peakind_curve_neg = find_peaks(-curve_smooth[:, fish], width=5)[0]
                    peak_curve_neg = curve_smooth[peakind_curve_neg, fish]

                    peakinds_curve = np.hstack((peakind_curve_pos, peakind_curve_neg))
                    peaks_curve = abs(np.hstack((peak_curve_pos, peak_curve_neg)))

                    I = np.argsort(peakinds_curve)
                    peakinds_curve = peakinds_curve[I]
                    peaks_curve = peaks_curve[I]

                    # plt.plot(curve_smooth[:,fish])
                    # plt.plot(peakinds_curve, peaks_curve, 'x')

                    # find the first peak the crosses the curvature threshold

                    if stim_given[trial] == 2:
                        curve_thresh = CBendThresh
                    else:
                        curve_thresh = OBendThresh

                    obend_peaks = np.where(peaks_curve > curve_thresh)[0]

                    # max curvature exibited during movie
                    max_curve = np.max(abs(curve[:, fish]))

                    # now get the kinematic aspects of the response
                    if len(obend_peaks) > 0:
                        start_o = np.nan
                        end_o = np.nan
                        obend_happened[fish] = 1
                        obend_peak = obend_peaks[0]
                        obend_peak_ind = peakinds_curve[obend_peak]
                        obend_peak_val = curve[obend_peak_ind, fish]

                        # determine where the fish is not moving based on speed trace and curvature trace being below a threshold after smoothing
                        not_moving = np.where(
                            (uniform_filter1d(speed[:, fish], 5, mode="nearest") < 0.3)
                            & (
                                uniform_filter1d(
                                    abs(curve_smooth[:, fish]), 5, mode="nearest"
                                )
                                < 0.3
                            )
                        )[0]

                        still_before = not_moving[not_moving < obend_peak_ind]

                        # if we cant find the start, stop analysis here
                        if len(still_before) > 0:
                            start_o = still_before[-1]

                            obend_start[fish] = start_o * 1000 / frame_rate

                            # get the angular velocity of the C1 movement from initiation to peak in degrees per msec
                            obend_ang_vel[fish] = np.rad2deg(
                                obend_peak_val
                                / (1000 / frame_rate * (obend_peak_ind - start_o))
                            )

                            # use when the speed and curvature returns to near 0 as the end of the movement to find end of moevement

                            still_after = not_moving[not_moving > obend_peak_ind]

                            # if we cant find the end, the movie cut of the end of the movement. can do this downstream analysis
                            if len(still_after) > 0:
                                end_o = still_after[0]

                                obend_dur[fish] = (end_o - start_o) * 1000 / frame_rate
                                obend_disp[fish] = (
                                    np.sqrt(
                                        np.square(
                                            x_coors[start_o, fish] - x_coors[end_o, fish]
                                        )
                                        + np.square(
                                            y_coors[start_o, fish] - y_coors[end_o, fish]
                                        )
                                    )
                                    * camera_rez
                                )
                                obend_dorient[fish] = np.rad2deg(
                                    HabTrackFunctions.subtract_angles(
                                        orientations.T[end_o, fish],
                                        orientations.T[start_o, fish],
                                    )
                                )
                                obend_max_curve[fish] = np.rad2deg(
                                    np.max(abs(curve[start_o:end_o, fish]))
                                )

                                # if obend_disp[fish] < 3: # fish needs to move at least 3 pixels, or else assume its a tracking error
                                #     fish_not_tracked[fish] = 1

                                # determine if this is a "multibend o bend" based on if the local minima after the C1 peak is below 0 (normal o-bend) or above 0 (multibend obend)
                                peak_curve = curve[peakinds_curve[obend_peak], fish]
                                if len(peakinds_curve) > (obend_peak + 1):
                                    trough_curve = curve[
                                        peakinds_curve[obend_peak + 1], fish
                                    ]
                                    obend_multibend[fish] = np.sign(peak_curve) == np.sign(
                                        trough_curve
                                    )
                                    # use the difference between peak and trough as c1 length
                                    obend_c1len[fish] = (
                                        (
                                            peakinds_curve[obend_peak + 1]
                                            - peakinds_curve[obend_peak]
                                        )
                                        * 1000
                                        / frame_rate
                                    )

                                # now look for a second O-bend
                                if max(peakinds_curve[obend_peaks]) > end_o:
                                    obend_second_counter[fish] = 1
                                else:
                                    obend_second_counter[fish] = 0

                        # else:
                        #     fish_not_tracked[fish] = 1

                    else:
                        obend_happened[fish] = 0

                obend_happened[fish_not_tracked] = np.nan
                track_data["Probability_Of_Response"][
                    trial, :
                ] = obend_happened  # 0 or 1, probability
                obend_start[fish_not_tracked] = np.nan
                track_data["Latency_(msec)"][trial, :] = obend_start  # already in msec
                obend_second_counter[fish_not_tracked] = np.nan
                track_data["Proportion_of_Double_Responses"][
                    trial, :
                ] = obend_second_counter  # 0 or 1, probability
                obend_dur[fish_not_tracked] = np.nan
                track_data["Movement_Duration_(msec)"][
                    trial, :
                ] = obend_dur  # units are in msec
                obend_disp[fish_not_tracked] = np.nan
                track_data["Displacement_(mm)"][trial, :] = obend_disp  # in units of mm
                obend_max_curve[fish_not_tracked] = np.nan
                track_data["Bend_Amplitude_(deg)"][
                    trial, :
                ] = obend_max_curve  # in units of degrees
                obend_dorient[fish_not_tracked] = np.nan
                track_data["Reorientation_(deg)"][
                    trial, :
                ] = obend_dorient  # in units of degrees
                obend_multibend[fish_not_tracked] = np.nan
                track_data["Proportion_of_Compound_Responses"][
                    trial, :
                ] = obend_multibend  # 0 or 1, probability
                obend_c1len[fish_not_tracked] = np.nan
                track_data["C1_Bend_Duration_(msec)"][
                    trial, :
                ] = obend_c1len  # units are in msec
                obend_ang_vel[fish_not_tracked] = np.nan
                track_data["C1_Angular_Velocity_(deg/msec)"][
                    trial, :
                ] = obend_ang_vel  # units are degrees per msec

            stim_times = np.array(track_data["TiffFrameInds"])
            stim_times = stim_times - stim_times[0]
            stim_times = stim_times / (frame_rate * 60 * 60)
            track_data["stim_times"] = stim_times

            os.chdir(graph_dir)

            save_name = (
                trial_file[: trial_file.find("_burst_frame_")] + data_analysis_suffix
            )
            with open(save_name, "wb") as f:
                pickle.dump(track_data, f)

            print("Saved data to: " + save_name)

    # % Combine data from all experiments into one file

    analyzed_pkls = glob.glob(os.path.realpath(root_dir + "\\*\\*response_data.pkl"))

    track_data_combined = {}
    n_loaded = 0
    n_files = len(analyzed_pkls)
    for k, track_name in enumerate(analyzed_pkls):
        exp_date = track_name.split(os.sep)[-2].split("_")[0]
        with open(track_name, "rb") as f:
            track_data = pickle.load(f)
            n_groups = len(track_data["rois"])

            if k == 0:
                track_data_combined = track_data.copy()
                track_data_combined["exp_date"] = []
                for gr in range(n_groups):
                    track_data_combined["exp_date"].append(exp_date)

            else:
                comps = list(track_data_combined.keys())[:10]
                for comp in comps:
                    comp_data = track_data[comp]
                    n_stim, n_fish = comp_data.shape
                    if n_stim == track_data_combined[comp].shape[0]:
                        track_data_combined[comp] = np.hstack(
                            (track_data_combined[comp], comp_data)
                        )
                    else:  # in some experiments we dont have the full stimulus run, will fill these up with NAN
                        comp_data_expand = np.full(
                            (track_data_combined[comp].shape[0], n_fish), np.nan
                        )
                        comp_data_expand[:n_stim, :] = comp_data
                        track_data_combined[comp] = np.hstack(
                            (track_data_combined[comp], comp_data_expand)
                        )

                for gr in range(n_groups):
                    track_data_combined["rois"].append(track_data["rois"][gr] + n_loaded)
                    track_data_combined["names"].append(track_data["names"][gr])
                    track_data_combined["plates"].append(track_data["plates"][gr])
                    track_data_combined["exp_date"].append(exp_date)

            n_rois = track_data["Probability_Of_Response"].shape[1]
            n_loaded += n_rois
            print(n_loaded)

    output_file_IDs = os.path.join(root_dir, "combined_data_IDs.csv")
    with open(output_file_IDs, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Name_ExperimentDate_Plate", "ROI"])
        for index, (roi, exp_date, plate, name) in enumerate(
            zip(
                track_data_combined["rois"],
                track_data_combined["exp_date"],
                track_data_combined["plates"],
                track_data_combined["names"],
            )
        ):
            writer.writerow(
                [index, str(name) + "_" + str(exp_date) + "_plate" + str(plate), roi]
            )

    track_file_combined = os.path.join(root_dir, "track_data_combined.pkl")
    with open(track_file_combined, "wb") as f:
        pickle.dump(track_data_combined, f)

# % load the pre-analyzed datasets to analyze individual experiments

track_file_combined = os.path.join(root_dir, "track_data_combined.pkl")
with open(track_file_combined, "rb") as f:
    track_data_combined = pickle.load(f)

    # Load combined_data_IDs.csv
combined_data_ids_path = os.path.join(root_dir, "combined_data_IDs.csv")
combined_data_ids = pd.read_csv(combined_data_ids_path)
combined_names = combined_data_ids["Name_ExperimentDate_Plate"].values

trained_response_stim = []
train_stim_block = 15
for i in range(n_blocks):
    trained_response_stim = np.hstack(
        (
            trained_response_stim,
            np.arange(
                i * n_stim_blocks + train_stim_block, i * n_stim_blocks + n_stim_blocks
            ),
        )
    )


stim_epochs = [
    np.arange(0, n_init),
    np.arange(n_init, n_stim_blocks),
    # np.arange(n_stim_blocks, n_stim_blocks * 2),
    # np.arange(n_stim_blocks * 2, n_stim_blocks * 3),
    # np.arange(n_stim_blocks * 3, n_stim_blocks * 4),
    np.array(trained_response_stim, dtype=int),
    # np.arange(n_stim_blocks * 4),
    np.where(track_data_combined["stim_given"] == 2)[0],
    np.where(track_data_combined["stim_given"] == 3)[0]
]

epoch_names = [
    "Naive Response",
    "Block 1",
    # "Block 2",
    # "Block 3",
    # "Block 4",
    "Trained Response",
    # "All DF Responses",
    "Acoustic Responses",
    "Block 5",
]
# %
# % generate color vector to be used in plots
p = gb.generate_palette(len(combined_names)+1)
col_vec = gb.convert_palette_to_rgb(p)
col_vec = list(np.array(col_vec[1:], dtype=float) / 255)

# define functions

def get_group_indexes(search_names, names_list, print_finds=True):
    indexes = []
    matched_names = []
    for i, name in enumerate(search_names):
        for j, pattern in enumerate(names_list):
            if name in pattern:
                if pattern not in matched_names:
                    indexes.append(j)
                    matched_names.append(pattern)
                    if print_finds:
                        print(f"Found index: {j}, Name: {pattern}")
                else:
                    if print_finds:
                        print(f"Duplicate found and ignored: {pattern}")

    if not indexes:
        print("No matches found for the given search patterns.")

    return indexes, matched_names


def get_rois(group_names, print_finds=True):
    group_indexes = []
    found_names = []
    plot_rois = []
    for gr in range(group_names.shape[1]):
        if print_finds:
            print("Group = " + group_categories[gr])
        index, names = get_group_indexes(
            group_names[:, gr], combined_names, print_finds
        )
        group_indexes.append(index)
        found_names.append(names)
        plot_rois.append(
            HabTrackFunctions.get_all_matching_ROIS(index, track_data_combined)
        )

    return plot_rois


def plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec,
    plot_cumdiff=True,
    cum_diff_components=np.arange(8),
):
    n_groups = group_names.shape[1]
    graph_folder = HabTrackFunctions.make_graph_folder(exp_string, root_dir)
    os.chdir(graph_folder)

    plot_rois = get_rois(group_names)
    HabTrackFunctions.plot_burst_responses(
        track_data_combined,
        group_categories,
        plot_rois,
        col_vec,
        exp_string,
        smooth_window=15,
        plot_taps=True,
        plot_retest=True,
        stim_times=track_data_combined["stim_times"],
    )

    HabTrackFunctions.plot_means_epoch(
        track_data_combined,
        group_categories,
        plot_rois,
        stim_epochs,
        epoch_names,
        exp_string,
        col_vec,
        components_to_plot=np.arange(10),
    )
    if plot_cumdiff:
        if n_groups > 2:
            for comp in list(combinations(np.arange(n_groups), 2)):
                plot_rois_comp = get_rois(group_names[:, comp], print_finds=False)
                group_categories_comp = []
                for gr in comp:
                    group_categories_comp.append(group_categories[gr])
                HabTrackFunctions.plot_cum_diff(
                    track_data_combined,
                    group_categories_comp,
                    plot_rois_comp,
                    exp_string,
                    components_to_plot=cum_diff_components,
                    n_boots=2000,
                    n_norm=n_init,
                )
        else:
            HabTrackFunctions.plot_cum_diff(
                track_data_combined,
                group_categories,
                plot_rois,
                exp_string,
                components_to_plot=cum_diff_components,
                n_boots=2000,
                n_norm=n_init,
            )


# %% plot mutant vs hets and WT
importlib.reload(HabTrackFunctions)
exp_string = os.path.join(graph_dir, "Melatonin_vs_DMSO_hetsAndWTs")


group_categories = ["DMSO", "Melatonin"]

group_names = np.array(
    [
        ["ba1 WT/Het 0.1% DMSO_20211019_plate0", "ba1 WT/Het 1uM Mela_20211019_plate0",],  
        ["ba2 WT/Het 0.1% DMSO_20211019_plate1", "ba2 WT/Het 1uM Mela_20211019_plate1",],  
        ["ba1 WT/Het 0.1% DMSO_20211020_plate0", "ba1 WT/Het 1uM Mela_20211020_plate0",],  
        ["ba2 WT/Het 0.1% DMSO_20211020_plate1", "ba2 WT/Het 1uM Mela_20211020_plate1",],  
        ["aa1 WT/Het 0.1% DMSO_20220117_plate0", "aa1 WT/Het melatonin 1µM_20220117_plate0",],  
        ["aa1 WT/Het 0.1% DMSO_20220117_plate1", "aa1 WT/Het melatonin 1µM_20220117_plate1",],  
        ["c4 WT/Het 0.1% DMSO_20220203_plate0", "c4 WT/Het melatonin 1µM_20220203_plate0",],  
        ["ab1 WT 0.1% DMSO_20220704_plate0", "ab1 WT 1µM Melatonin_20220704_plate0",],  
        ["ab1 Het 0.1% DMSO_20220704_plate0", "ab1 Het 1µM Melatonin_20220704_plate0",],  
        ["ba2/C4 WT/Het 0.1% DMSO_20220707_plate0", "ba2/C4 WT/Het 0.1% DMSO_20220707_plate0",],  
        ["aa2+/?; al2+/? ; DMSO_20221129_plate0", "aa2+/?; al2+/? ; Mel_20221129_plate0",],  
        ["aa2+/?; al2+/? ; DMSO_20221129_plate1", "aa2+/?; al2+/? ; Mel_20221129_plate1",],  
        ["aa2+/?; ab2+/? ; DMSO_20221130_plate0", "aa2+/?; ab2+/? ; Mel_20221130_plate0",],  
        ["aa2+/?; ab2+/? ; DMSO_20221130_plate1", "aa2+/?; ab2+/? ; Mel_20221130_plate1",],  
        ["aa2+/?; al2+/? ; DMSO_20230110_plate1", "aa2+/?; al2+/? ; Mel_20230110_plate1",],  
        ["aa2+/?; ab2+/? ; DMSO_20230123_plate0", "aa2+/?; ab2+/? ; Mel_20230123_plate0",],  
        ["aa2+/?; ab2+/? ; DMSO_20230123_plate1", "aa2+/?; ab2+/? ; Mel_20230123_plate1",],  
        ["ab1 +/? DMSO_20230620_plate0", "ab1+/? 1µM Melatonin_20230620_plate0",],  
        # ["", "",],  
    ]

)


plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=True,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)
#%%
exp_string = os.path.join(graph_dir, "AA1 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1aa^{+/?}}$",
    r"Melatonin, $\it{mtnr1aa^{+/?}}$",
    r"DMSO, $\it{mtnr1aa^{-/-}}$",
    r"Melatonin, $\it{mtnr1aa^{-/-}}$",
    ]

group_names = np.array(
    [
        ['aa1 WT/Het 0.1% DMSO_20220117_plate0',
         'aa1 WT/Het melatonin 1µM_20220117_plate0',
         'aa1 mutant 0.1% DMSO_20220117_plate0',
         'aa1 mutant melatonin 1µM_20220117_plate0',],  
        
        ['aa1 WT/Het 0.1% DMSO_20220117_plate1',
         'aa1 WT/Het melatonin 1µM_20220117_plate1',
         'aa1 mutant 0.1% DMSO_20220117_plate1',
         'aa1 mutant melatonin 1µM_20220117_plate1',],  
        
        
        # ['',
        #  '',
        #  '',
        #  '',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)
#%%

exp_string = os.path.join(graph_dir, "AA2 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1aa^{+/?}}$",
    r"Melatonin, $\it{mtnr1aa^{+/?}}$",
    r"DMSO, $\it{mtnr1aa^{-/-}}$",
    r"Melatonin, $\it{mtnr1aa^{-/-}}$",
    ]

group_names = np.array(
    [
        ['aa1 WT/Het 0.1% DMSO_20220117_plate0',
         'aa1 WT/Het melatonin 1µM_20220117_plate0',
         'aa1 mutant 0.1% DMSO_20220117_plate0',
         'aa1 mutant melatonin 1µM_20220117_plate0',],  
        
        ['aa1 WT/Het 0.1% DMSO_20220117_plate1',
         'aa1 WT/Het melatonin 1µM_20220117_plate1',
         'aa1 mutant 0.1% DMSO_20220117_plate1',
         'aa1 mutant melatonin 1µM_20220117_plate1',], 
        
        ['aa2 WT/Het DMSO_20230110_plate1',
         'aa2 WT/Het 1µM Melatonin_20230110_plate1',
         'aa2 mutant DMSO_20230110_plate1',
         'aa2 mutant 1µM Melatonin_20230110_plate1',], 
        
        ['aa2 WT/Het DMSO_20230123_plate0',
         'aa2 WT/Het 1µM Melatonin_20230123_plate0',
         'aa2 mutant DMSO_20230123_plate0',
         'aa2 mutant 1µM Melatonin_20230123_plate0',], 
        
        ['aa2 WT/Het DMSO_20230123_plate1',
         'aa2 WT/Het 1µM Melatonin_20230123_plate1',
         'aa2 mutant DMSO_20230123_plate1',
         'aa2 mutant 1µM Melatonin_20230123_plate1',], 
        
        ['aa2 WT/Het DMSO_20221129_plate0',
         'aa2 WT/Het 1µM Melatonin_20221129_plate0',
         'aa2 mutant DMSO_20221129_plate0',
         'aa2 mutant 1µM Melatonin_20221129_plate0',],  
        
        ['aa2 WT/Het DMSO_20221129_plate1',
         'aa2 WT/Het 1µM Melatonin_20221129_plate1',
         'aa2 mutant DMSO_20221129_plate1',
         'aa2 mutant 1µM Melatonin_20221129_plate1',],  
        
        ['aa2 WT/Het DMSO_20221130_plate0',
         'aa2 WT/Het 1µM Melatonin_20221130_plate0',
         'aa2 mutant DMSO_20221130_plate0',
         'aa2 mutant 1µM Melatonin_20221130_plate0',], 
        
        ['aa2 WT/Het DMSO_20221130_plate1',
         'aa2 WT/Het 1µM Melatonin_20221130_plate1',
         'aa2 mutant DMSO_20221130_plate1',
         'aa2 mutant 1µM Melatonin_20221130_plate1',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=True,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%

exp_string = os.path.join(graph_dir, "AA Mutants Combined")


group_categories = [
    r"DMSO, $\it{mtnr1aa^{+/?}}$",
    r"Melatonin, $\it{mtnr1aa^{+/?}}$",
    r"DMSO, $\it{mtnr1aa^{-/-}}$",
    r"Melatonin, $\it{mtnr1aa^{-/-}}$",
    ]

group_names = np.array(
    [
        ['aa2 WT/Het DMSO_20230110_plate1',
         'aa2 WT/Het 1µM Melatonin_20230110_plate1',
         'aa2 mutant DMSO_20230110_plate1',
         'aa2 mutant 1µM Melatonin_20230110_plate1',], 
        
        ['aa2 WT/Het DMSO_20230123_plate0',
         'aa2 WT/Het 1µM Melatonin_20230123_plate0',
         'aa2 mutant DMSO_20230123_plate0',
         'aa2 mutant 1µM Melatonin_20230123_plate0',], 
        
        ['aa2 WT/Het DMSO_20230123_plate1',
         'aa2 WT/Het 1µM Melatonin_20230123_plate1',
         'aa2 mutant DMSO_20230123_plate1',
         'aa2 mutant 1µM Melatonin_20230123_plate1',], 
        
        ['aa2 WT/Het DMSO_20221129_plate0',
         'aa2 WT/Het 1µM Melatonin_20221129_plate0',
         'aa2 mutant DMSO_20221129_plate0',
         'aa2 mutant 1µM Melatonin_20221129_plate0',],  
        
        ['aa2 WT/Het DMSO_20221129_plate1',
         'aa2 WT/Het 1µM Melatonin_20221129_plate1',
         'aa2 mutant DMSO_20221129_plate1',
         'aa2 mutant 1µM Melatonin_20221129_plate1',],  
        
        ['aa2 WT/Het DMSO_20221130_plate0',
         'aa2 WT/Het 1µM Melatonin_20221130_plate0',
         'aa2 mutant DMSO_20221130_plate0',
         'aa2 mutant 1µM Melatonin_20221130_plate0',], 
        
        ['aa2 WT/Het DMSO_20221130_plate1',
         'aa2 WT/Het 1µM Melatonin_20221130_plate1',
         'aa2 mutant DMSO_20221130_plate1',
         'aa2 mutant 1µM Melatonin_20221130_plate1',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=True,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)


#%%
exp_string = os.path.join(graph_dir, "AL Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1al^{+/?}}$",
    r"Melatonin, $\it{mtnr1al^{+/?}}$",
    r"DMSO, $\it{mtnr1al^{-/-}}$",
    r"Melatonin, $\it{mtnr1al^{-/-}}$",
    ]

group_names = np.array(
    [      
        ['al2 WT/Het 0.1% DMSO_20220830_plate0',
         'al2 WT/Het 1µM Melatonin_20220830_plate0',
         'al2 mutant 0.1% DMSO_20220830_plate0',
         'al2 mutant 1µM Melatonin_20220830_plate0',], 
        
        ['al2 WT/Het 0.1% DMSO_20221129_plate0',
         'al2 WT/Het 1µM Melatonin_20221129_plate0',
         'al2 mutant 0.1% DMSO_20221129_plate0',
         'al2 mutant 1µM Melatonin_20221129_plate0',], 
        
        ['al2 WT/Het 0.1% DMSO_20221129_plate1',
         'al2 WT/Het 1µM Melatonin_20221129_plate1',
         'al2 mutant 0.1% DMSO_20221129_plate1',
         'al2 mutant 1µM Melatonin_20221129_plate1',], 
        
        ['al2 WT/Het 0.1% DMSO_20230110_plate1',
         'al2 WT/Het 1µM Melatonin_20230110_plate1',
         'al2 mutant 0.1% DMSO_20230110_plate1',
         'al2 mutant 1µM Melatonin_20230110_plate1',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)


#%%

exp_string = os.path.join(graph_dir, "AB1 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1ab^{+/?}}$",
    r"Melatonin, $\it{mtnr1ab^{+/?}}$",
    r"DMSO, $\it{mtnr1ab^{-/-}}$",
    r"Melatonin, $\it{mtnr1ab^{-/-}}$",
    ]

group_names = np.array(
    [      
        
        ['ab1 +/? DMSO_20230620_plate0',
         'ab1+/? 1µM Melatonin_20230620_plate0',
         'ab1 -/- DMSO_20230620_plate0',
         'ab1-/- 1µM Melatonin_20230620_plate0',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
        
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)
#%%



exp_string = os.path.join(graph_dir, "AB2 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1ab^{+/?}}$",
    r"Melatonin, $\it{mtnr1ab^{+/?}}$",
    r"DMSO, $\it{mtnr1ab^{-/-}}$",
    r"Melatonin, $\it{mtnr1ab^{-/-}}$",
    ]

group_names = np.array(
    [      
        
        ['ab2 WT/Het 0.1% DMSO_20220830_plate0',
         'ab2 WT/Het 1µM Melatonin_20220830_plate0',
         'ab2 mutant 0.1% DMSO_20220830_plate0',
         'ab2 mutant 1µM Melatonin_20220830_plate0',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
        
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%

exp_string = os.path.join(graph_dir, "AB Mutants Combined")


group_categories = [
    r"DMSO, $\it{mtnr1ab^{+/?}}$",
    r"Melatonin, $\it{mtnr1ab^{+/?}}$",
    r"DMSO, $\it{mtnr1ab^{-/-}}$",
    r"Melatonin, $\it{mtnr1ab^{-/-}}$",
    ]

group_names = np.array(
    [      

        ['ab1 +/? DMSO_20230620_plate0',
         'ab1+/? 1µM Melatonin_20230620_plate0',
         'ab1 -/- DMSO_20230620_plate0',
         'ab1-/- 1µM Melatonin_20230620_plate0',], 
        
        ['ab2 WT/Het 0.1% DMSO_20220830_plate0',
         'ab2 WT/Het 1µM Melatonin_20220830_plate0',
         'ab2 mutant 0.1% DMSO_20220830_plate0',
         'ab2 mutant 1µM Melatonin_20220830_plate0',], 
        
        # ['',
        #  '',
        #  '',
        #  '',], 
        
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%



exp_string = os.path.join(graph_dir, "BA1 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1ba^{+/?}}$",
    r"Melatonin, $\it{mtnr1ba^{+/?}}$",
    r"DMSO, $\it{mtnr1ba^{-/-}}$",
    r"Melatonin, $\it{mtnr1ba^{-/-}}$",
    ]

group_names = np.array(
    [      

        ['ba1 WT/Het 0.1% DMSO_20211019_plate0',
         'ba1 WT/Het 1uM Mela_20211019_plate0',
         'ba1 Mutant DMSO_20211019_plate0',
         'ba1 Mutant 1uM Mela_20211019_plate0',], 
        
        ['ba1 WT/Het 0.1% DMSO_20211020_plate0',
         'ba1 WT/Het 1uM Mela_20211020_plate0',
         'ba1 Mutant DMSO_20211020_plate0',
         'ba1 Mutant 1uM Mela_20211020_plate0',], 
        
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)



#%%
exp_string = os.path.join(graph_dir, "BA2 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1ba^{+/?}}$",
    r"Melatonin, $\it{mtnr1ba^{+/?}}$",
    r"DMSO, $\it{mtnr1ba^{-/-}}$",
    r"Melatonin, $\it{mtnr1ba^{-/-}}$",
    ]

group_names = np.array(
    [      

        ['ba2 WT/Het 0.1% DMSO_20211019_plate1',
         'ba2 WT/Het 1uM Mela_20211019_plate1',
         'ba2 Mutant DMSO_20211019_plate1',
         'ba2 Mutant 1uM Mela_20211019_plate1',], 
        
        ['ba2 WT/Het 0.1% DMSO_20211020_plate1',
         'ba2 WT/Het 1uM Mela_20211020_plate1',
         'ba2 Mutant DMSO_20211020_plate1',
         'ba2 Mutant 1uM Mela_20211020_plate1',], 
        
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%
exp_string = os.path.join(graph_dir, "BA Mutants Combined")


group_categories = [
    r"DMSO, $\it{mtnr1ba^{+/?}}$",
    r"Melatonin, $\it{mtnr1ba^{+/?}}$",
    r"DMSO, $\it{mtnr1ba^{-/-}}$",
    r"Melatonin, $\it{mtnr1ba^{-/-}}$",
    ]

group_names = np.array(
    [     
     
        ['ba1 WT/Het 0.1% DMSO_20211019_plate0',
         'ba1 WT/Het 1uM Mela_20211019_plate0',
         'ba1 Mutant DMSO_20211019_plate0',
         'ba1 Mutant 1uM Mela_20211019_plate0',], 
        
        ['ba1 WT/Het 0.1% DMSO_20211020_plate0',
         'ba1 WT/Het 1uM Mela_20211020_plate0',
         'ba1 Mutant DMSO_20211020_plate0',
         'ba1 Mutant 1uM Mela_20211020_plate0',],  

        ['ba2 WT/Het 0.1% DMSO_20211019_plate1',
         'ba2 WT/Het 1uM Mela_20211019_plate1',
         'ba2 Mutant DMSO_20211019_plate1',
         'ba2 Mutant 1uM Mela_20211019_plate1',], 
        
        ['ba2 WT/Het 0.1% DMSO_20211020_plate1',
         'ba2 WT/Het 1uM Mela_20211020_plate1',
         'ba2 Mutant DMSO_20211020_plate1',
         'ba2 Mutant 1uM Mela_20211020_plate1',], 
        
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%
exp_string = os.path.join(graph_dir, "C4 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1c^{+/?}}$",
    r"Melatonin, $\it{mtnr1c^{+/?}}$",
    r"DMSO, $\it{mtnr1c^{-/-}}$",
    r"Melatonin, $\it{mtnr1c^{-/-}}$",
    ]

group_names = np.array(
    [      


        ['c4 WT/Het 0.1% DMSO_20220203_plate0',
         'c WT/Het melatonin 1µM_20220203_plate0',
         'c mutant 0.1% DMSO_20220203_plate0',
         'c mutant melatonin 1µM_20220203_plate0',], 

        ['ba2/C4 WT/Het 0.1% DMSO_20220707_plate0',
         'ba2/C4 WT/Het 1µM Melatonin_20220707_plate0',
         'ba2/C4 mutant 0.1% DMSO_20220707_plate0',
         'ba2/C4 mutant1µM Melatonin_20220707_plate0',], 
        
        # ['',
        #  '',
        #  '',
        #  '',],         
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%
exp_string = os.path.join(graph_dir, "BB5 Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1bb^{+/?}}$",
    r"Melatonin, $\it{mtnr1bb^{+/?}}$",
    r"DMSO, $\it{mtnr1bb^{-/-}}$",
    r"Melatonin, $\it{mtnr1bb^{-/-}}$",
    ]

group_names = np.array(
    [      


        ['wt/het DMSO 0.1% _20260422_plate0',
         'wt/het Melatonin 1uM_20260422_plate0',
         'mutant DMSO 0.1% _20260422_plate0',
         'mutant Melatonin 1uM_20260422_plate0',], 

        
        # ['',
        #  '',
        #  '',
        #  '',],         
    
    ]
        
)
plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec[: group_names.shape[1]],
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)


#%%
exp_string = os.path.join(graph_dir, "AA AL Mutants")


group_categories = [
    r"DMSO, $\it{mtnr1aa^{+/?}; mtnr1al^{+/?}}$",
    r"Melatonin, $\it{mtnr1aa^{+/?}; mtnr1al^{+/?}}$",
    # r"DMSO, $\it{mtnr1aa^{-/-}; mtnr1al^{-/-}}$",
    r"Melatonin, $\it{mtnr1aa^{-/-}; mtnr1al^{-/-}}$",
    ]

group_names = np.array(
    [      

        ['aa2+/?; al2+/? ; DMSO_20221129_plate0',
         'aa2+/?; al2+/? ; Mel_20221129_plate0',

         'aa2-/-;al2-/-; Mel_20221129_plate0',], 
        
        ['aa2+/?; al2+/? ; DMSO_20221129_plate1',
         'aa2+/?; al2+/? ; Mel_20221129_plate1',

         'aa2-/-;al2-/-; Mel_20221129_plate1',],    
        
        ['aa2+/?; ab2+/? ; DMSO_20221130_plate0',
         'aa2+/?; ab2+/? ; Mel_20221130_plate0',

         'aa2-/-;ab2-/-; Mel_20221130_plate0',], 
        
        ['aa2+/?; ab2+/? ; DMSO_20221130_plate1',
         'aa2+/?; ab2+/? ; Mel_20221130_plate1',

         'aa2-/-;ab2-/-; Mel_20221130_plate1',],      

        ['aa2+/?; al2+/? ; DMSO_20230110_plate1',
         'aa2+/?; al2+/? ; Mel_20230110_plate1',

         'aa2-/-;al2-/-; Mel_20230110_plate1',], 



    ]


)

plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    list(np.array(col_vec)[[0,1,4]]),
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)


#%%

#%%
exp_string = os.path.join(graph_dir, "Subjective Night vs Day")


group_categories = [
    r"Normal",
    r"Shifted Light Cycle",
    ]

group_names = np.array(
    [      

        ['Normal aanat 1 +/? 2 +/?_20230320_plate0',
         'Shifted aanat 1 +/? 2 +/?_20230320_plate0',], 
        
        ['Normal aanat 1 +/? 2 +/?_20230321_plate0',
         'Shifted aanat 1 +/? 2 +/?_20230321_plate0',], 
        
        ['Normal aanat 1 +/? 2 +/?_20230418_plate0',
         'Shifted aanat 1 +/? 2 +/?_20230418_plate0',], 


        ['Normal aanat 1 +/? 2 +/?_20230504_plate0',
         'Shifted aanat 1 +/? 2 +/?_20230504_plate0',], 
        
    ]


)

plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    list(np.array(col_vec)[[0,1,4]]),
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%

exp_string = os.path.join(graph_dir, "AANAT Mutants Subjective Night")


group_categories = [
    r"Shifted Light Cycle; $\it{aanat1^{+/?}; aanat2^{+/?}}$",
    
    r"Shifted Light Cycle; $\it{aanat1^{-/-}; aanat2^{+/?}}$",
    
    r"Shifted Light Cycle; $\it{aanat1^{+/?}; aanat2^{-/-}}$",
    
    r"Shifted Light Cycle; $\it{aanat1^{-/-}; aanat2^{-/-}}$",
    
    ]

group_names = np.array(
    [      

        ['Shifted aanat 1 +/? 2 +/?_20230320_plate0', 
         'Shifted aanat 1 -/- 2 +/-_20230320_plate0',
        'Shifted aanat 1 +/- 2 -/-_20230320_plate0',
        'Shifted aanat 1 -/- 2 -/-_20230320_plate0'
        ],
        
        ['Shifted aanat 1 +/? 2 +/?_20230321_plate0', 
         'Shifted aanat 1 -/- 2 +/?_20230321_plate0',
        'Shifted aanat 1 +/? 2 -/-_20230321_plate0',
        'Shifted aanat 1 -/- 2 -/-_20230321_plate0'
        ],
        
        ['Shifted aanat 1 +/? 2 -/-_20230418_plate0', 
        'Shifted aanat 1 -/- 2 +/?_20230418_plate0',
        'Shifted aanat 1 +/? 2 -/-_20230418_plate0',
        'Shifted aanat 1 -/- 2 -/-_20230418_plate0'
        ],
        
        ['Shifted aanat 1 +/? 2 +/?_20230504_plate0', 
         'Shifted aanat 1 -/- 2 +/?_20230504_plate0',
        'Shifted aanat 1 +/? 2 -/-_20230504_plate0',
        'Shifted aanat 1 -/- 2 -/-_20230504_plate0'
        ],




    ]


)

plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec,
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)

#%%

exp_string = os.path.join(graph_dir, "AANAT Mutants Day")

group_categories = [
    r"Normal Light Cycle; $\it{aanat1^{+/?}; aanat2^{+/?}}$",
    
    r"Normal Light Cycle; $\it{aanat1^{-/-}; aanat2^{+/?}}$",
    
    r"Normal Light Cycle; $\it{aanat1^{+/?}; aanat2^{-/-}}$",
    
    r"Normal Light Cycle; $\it{aanat1^{-/-}; aanat2^{-/-}}$",
    
    ]

group_names = np.array(
    [      

        ['Normal aanat 1 +/? 2 +/?_20230320_plate0', 
         'Normal aanat 1 -/- 2 +/-_20230320_plate0',
        'Normal aanat 1 +/- 2 -/-_20230320_plate0',
        'Normal aanat 1 -/- 2 -/-_20230320_plate0'
        ],
        
        ['Normal aanat 1 +/? 2 +/?_20230321_plate0', 
         'Normal aanat 1 -/- 2 +/?_20230321_plate0',
        'Normal aanat 1 +/? 2 -/-_20230321_plate0',
        'Normal aanat 1 -/- 2 -/-_20230321_plate0'
        ],
        
        ['Normal aanat 1 +/? 2 -/-_20230418_plate0', 
        'Normal aanat 1 -/- 2 +/?_20230418_plate0',
        'Normal aanat 1 +/? 2 -/-_20230418_plate0',
        'Normal aanat 1 -/- 2 -/-_20230418_plate0'
        ],
        
        ['Normal aanat 1 +/? 2 +/?_20230504_plate0', 
         'Normal aanat 1 -/- 2 +/?_20230504_plate0',
        'Normal aanat 1 +/? 2 -/-_20230504_plate0',
        'Normal aanat 1 -/- 2 -/-_20230504_plate0'
        ],




    ]

)


plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_vec,
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)


#%%

exp_string = os.path.join(graph_dir, "AANAT Mutants Day and Night")


group_categories = [
    r"Normal Light Cycle; $\it{aanat1^{+/?}; aanat2^{+/?}}$", 
    r"Normal Light Cycle; $\it{aanat1^{-/-}; aanat2^{-/-}}$", 
    r"Shifted Light Cycle; $\it{aanat1^{+/?}; aanat2^{+/?}}$",   
    r"Shifted Light Cycle; $\it{aanat1^{-/-}; aanat2^{-/-}}$",
    ]

group_names = np.array(
    [      

        ['Normal aanat 1 +/? 2 +/?_20230320_plate0', 
         'Normal aanat 1 -/- 2 -/-_20230320_plate0',
         'Shifted aanat 1 +/? 2 +/?_20230320_plate0', 
         
         'Shifted aanat 1 -/- 2 -/-_20230320_plate0'

        ],
        
        ['Normal aanat 1 +/? 2 +/?_20230321_plate0', 
         'Normal aanat 1 -/- 2 -/-_20230321_plate0',
         'Shifted aanat 1 +/? 2 +/?_20230321_plate0', 
         
         'Shifted aanat 1 -/- 2 -/-_20230321_plate0'

        ],
        
        ['Normal aanat 1 +/? 2 -/-_20230418_plate0', 
         'Normal aanat 1 -/- 2 -/-_20230418_plate0',
         'Shifted aanat 1 +/? 2 -/-_20230418_plate0', 
         
         'Shifted aanat 1 -/- 2 -/-_20230418_plate0'

        ],
        
        ['Normal aanat 1 +/? 2 +/?_20230504_plate0', 
         'Normal aanat 1 -/- 2 -/-_20230504_plate0',
        'Shifted aanat 1 +/? 2 +/?_20230504_plate0', 
        
        'Shifted aanat 1 -/- 2 -/-_20230504_plate0'

        ],




    ]

)

col_nightday = [
    np.array([0.7, 0.549, 0.]),
    np.array([0.7, 0.549, 0.5]),
    np.array([0.1, 0.1, 0.1]),
    
    np.array([0.1, 0.1, 0.45]),
]

plot_bursts_and_epochs(
    exp_string,
    group_categories,
    group_names,
    col_nightday,
    plot_cumdiff=False,
    cum_diff_components=[0, 1, 2, 3, 4, 5, 6, 7],
)