import yaml
import sys
import os
import pickle
import numpy as np
from pytictoc import TicToc
from decimal import Decimal


pythonfiles_location = os.getcwd() + '/' #'/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/ParamSearch/'
results_folder_top = os.getcwd() + '/'#'/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Results/Aug24th/final_1/'
IQ_folder = '/Users/venkat/Documents/scisrs/Emanations/IQSamples/June16thCollect/LaptopConnectedToMonitorviaAdaptor/Laptop_MonitorViaAdaptor_25MHzSlice/'

from EmanationDetection_search import *


scenario_IQfolder_dict = {}
scenario_IQfolder_dict['Laptop_MonitorViaAdaptor'] = {}
scenario_IQfolder_dict['Laptop_MonitorViaAdaptor']['durationcapture_ms'] = '300'
scenario_list = ['Laptop_MonitorViaAdaptor']

# Split 200 Mhz capture into 8 logical slot indices: 0 to 7.
# for CF = 200, capture range is 100 to 300 MHz.
# if you want to detect emanation on 150 to 175 MHz, we give slot as 2.
CF_range_start = 200 # in MHz
freq_slot_start = 1
CF_range_end = 300 # in MHz
freq_slot_end = 2


freq_slot_range = np.arange(freq_slot_start, freq_slot_end, 1)
CF_range = np.arange(CF_range_start, CF_range_end, 200)

hyper_param = {}

# wt_range = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
try:
    os.mkdir(results_folder_top)
except OSError as error:
    print(error)

PSD_plot_param = {}
# we sometimes want to plot only a percentage of x-axis to get e zoomed version.
# we therefore take a percentage of the x-axis and zoom accordingly and make the plot.
PSD_plot_param['zoom_perc'] = [[4, 30, 100], [4, 30, 100], [4, 30, 100], [4, 30, 100]]
diffcolor_eachharmonic_flag = True  # when overlaying peaks on the PSD< we pick a different color for each of
PSD_plot_param['diffcolor_eachharmonic'] = diffcolor_eachharmonic_flag
spectrogram_flag, PSD_flag, Objfunc_ErrvsFreq, peaks_flag = True, True, True, True
plot_flags = [spectrogram_flag, PSD_flag, Objfunc_ErrvsFreq, peaks_flag]
cmap = 'viridis'
plot_dict = {'spectrogram': plot_flags[0],
             'PSD': plot_flags[1],
             'Objfunc_ErrvsFreq': plot_flags[2], \
             'peaks': plot_flags[3], 'cmap': cmap}
hyper_param['err_thresh_perc'] = 2
hyper_param['p1'] = 0.5
hyper_param['p2'] = 0.5
s_range = [-1]  # np.arange(-1,-10,-2)



########
######
def Iteration_perHyperParam(scenario_list, scenario_IQfolder_dict, CF_range, freq_slot_range, results_folder, \
                            results_folder_top, hyper_param_string, config_dict, plot_dict, PSD_plot_param):
    samprate = 200e6
    BW = 200  # Mhz
    results_dict = {}
    numtaps1 = 1000
    f_step1 = 25e6
    samprate_slice = f_step1
    kaiser_beta1 = 20
    trial_num = 0

    CF_slice_range = []
    for scenario in scenario_list:

        # duration_capture = scenario_IQfolder_dict[scenario]['durationcapture_ms']

        results_folder = results_folder + '/'
        try:
            os.mkdir(results_folder)
        except OSError as error:
            print(error)

        for CF in CF_range:
            for freq_slot in freq_slot_range:
                SF_freqslot, EF_freqslot = -samprate / 2 + (freq_slot) * f_step1, -samprate / 2 + (
                            freq_slot + 1) * f_step1
                CF_freqslot = (SF_freqslot + EF_freqslot) / 2
                # shift_freq = -1 * (SF_freqslot + EF_freqslot) / 2
                # cutoff1 = f_step1 / 2

                filename = 'Scenario_' + scenario + '_CF_' + str(int(CF + CF_freqslot / 1e6)) + 'MHz' + '.pkl'
                with open(IQ_folder +  filename, 'rb') as dict_file:
                    dict_IQ = pickle.load(dict_file)
                filt_dec_signal = dict_IQ['IQ']

                ####### SLicing/bandpass filtering via kaiser filter
                CF_p1_p2 = str(int(CF_freqslot / 1e6 + CF)) + hyper_param_string
                print("Scenario is: ", scenario)
                print(" CF: ", CF_freqslot + CF * 1e6)
                print("Range of frequencies: Start freq: ", CF + SF_freqslot / 1e6, ' MHz. End freq: ',
                      CF + EF_freqslot / 1e6, ' MHz.')
                data = {'iq': filt_dec_signal, 'sample_rate': samprate_slice, 'center_freq': CF_freqslot + CF * 1e6, \
                        'time_duration': trial_num, 'path': results_folder, \
                        'scenario': scenario, 'pythonfiles_location': pythonfiles_location, 'plot_dict': plot_dict, \
                        'CF_p1_p2': CF_p1_p2, 'PSD_plot_param': PSD_plot_param}

                t = TicToc()
                t.tic()
                dict_resultsval = EmanationDetection(data, config_dict)
                t.toc()
                CF_slice_range.append(int(data['center_freq'] / 1e6))

        # scenario = 'busy_conference_environ'
        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(results_folder + 'MeasuredPartials_divide_PitchEstimate_AcrossFreq.txt',
                  'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            # print('This message will be written to a file.')

            for CF_slice in CF_slice_range:  # np.arange(137, 1113, 25):
                results_file = results_folder  + 'Scenario_' + scenario + '_CF_' + str(
                    CF_slice) + 'MHz.pkl'

                with open(results_file, 'rb') as file:
                    resultval = pickle.load(file)
                file.close()
                for keyval in resultval['results'].keys():
                    components_relativefreq = resultval['results'][keyval]['components_relativefreq']
                    # We are picking the median SNR of the lowest 5 harmonics as the SNR for the entire harmonic series.
                    SNR_val = np.median(resultval['results'][keyval]['SNR'][0:5])
                    remainder = np.divide(components_relativefreq, keyval)
                    print("CF: ", CF_slice)
                    print("Pitch: ", keyval)
                    print("SNR: ", SNR_val)
                    print(remainder)
            sys.stdout = original_stdout

            # scenario = 'busy_conference_environ'
            original_stdout = sys.stdout  # Save a reference to the original standard output

            with open(results_folder + 'Pitch_SNR_acrossFreq.txt', 'w') as f:
                sys.stdout = f  # Change the standard output to the file we created.
                # print('This message will be written to a file.')

                for CF_slice in CF_slice_range:  # np.arange(137, 1113, 25):
                    results_file = results_folder +  'Scenario_' + scenario + '_CF_' + str(
                        CF_slice) + 'MHz.pkl'

                    with open(results_file, 'rb') as file:
                        resultval = pickle.load(file)
                    file.close()
                    for keyval in resultval['results'].keys():
                        # We are picking the median SNR of the lowest 5 harmonics as the SNR for the entire harmonic series.
                        SNR_val = np.median(resultval['results'][keyval]['SNR'][0:5])
                        print("CF: ", CF_slice, "MHz. Pitch: ", round(keyval, 2), ". SNR: ", SNR_val)
                sys.stdout = original_stdout

            ####### Write values for each scenario to a single file that sits top of scenario folders.
            ####### Values shud be just the total number of pitch values each scenario for that chosen value of hyper parameters
            original_stdout = sys.stdout  # Save a reference to the original standard output
            Results_file = results_folder + 'Num_EmanationsPerScenario.txt'
            if os.path.exists(Results_file):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not

            with open(Results_file, append_write) as f:
                sys.stdout = f  # Change the standard output to the file we created.
                # print('This message will be written to a file.')
                total_count_num_eman = 0
                for CF_slice in CF_slice_range:  # np.arange(137, 1113, 25):
                    results_file = results_folder + 'Scenario_' + scenario + '_CF_' + str(
                        CF_slice) + 'MHz.pkl'

                    with open(results_file, 'rb') as file:
                        resultval = pickle.load(file)
                    total_count_num_eman = total_count_num_eman + len(list(resultval['results'].keys()))
                print(hyper_param_string, ' ', scenario, ' is: ', total_count_num_eman)
                sys.stdout = original_stdout


################
########################################################################################################################
# UPDATE YAML FILE
########################################################################################################################
# Function to update values in the YAML file
def update_yaml_file(hyper_param, file_path="synapse_emanation_search.yaml"):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    # yaml_data['EstimateHarmonic']['s1'] = float(str(hyper_param['s1']))
    # yaml_data['EstimateHarmonic']['s2'] = float(str(hyper_param['s2']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][100000000000] = float(str(hyper_param['err_thresh_perc']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][1000] = float(str(hyper_param['err_thresh_perc']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][50000] = float(str(hyper_param['err_thresh_perc']))
    # Update the data
    yaml_data['EstimateHarmonic']['p_hh_1'] = float(str(hyper_param['p1']))
    yaml_data['EstimateHarmonic']['p_hh_2'] = float(str(hyper_param['p2']))
    yaml_data['EstimateHarmonic']['p_lh_1'] = float(str(hyper_param['p1']))
    yaml_data['EstimateHarmonic']['p_lh_2'] = float(str(hyper_param['p2']))
    print("yaml_data['EstimateHarmonic']: ", yaml_data['EstimateHarmonic'])
    return yaml_data


# for hyper_param['s1'] in s_range:
#     for hyper_param['s2'] in s_range:
        #         for hyper_param['wt'] in wt_range:
config_dict = update_yaml_file(hyper_param)

PSD_plot_param['dur_ensemble'] = [0.1, config_dict['EmanationDetection']['dur_ensemble'], 0.1, 0.1]

hyper_param_string = 'Results'#'s1_' + str(hyper_param['s1']) + 's2_' + str(hyper_param['s2'])
                     #+ 'p_' + str(hyper_param['p1']) + "_Errthresh_2"

results_folder = results_folder_top + hyper_param_string  # 'p1_'+str(hyper_param['p1']) + '_p2_'+str(hyper_param['p2'])
try:
    os.mkdir(results_folder)
except OSError as error:
    print(error)
with open(results_folder + '/' + 'config_dict.pkl', 'wb') as f:
    pickle.dump(config_dict, f)

# 'p1_'+str(hyper_param['p1']) + '_p2_'+str(hyper_param['p2'])
Iteration_perHyperParam(scenario_list, scenario_IQfolder_dict, CF_range, freq_slot_range, results_folder, \
                        results_folder_top, hyper_param_string, config_dict, plot_dict, PSD_plot_param)