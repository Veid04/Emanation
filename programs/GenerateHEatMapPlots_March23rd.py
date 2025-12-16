import yaml
import sys
import os
import pickle
import numpy as np
# from pytictoc import TicToc
from decimal import Decimal
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np

IQ_25MHz_flag = True
device = 'laptop'  # 'server'
# Aug9thp1_p2searchp1_0.1_p2_0.3
if device == 'laptop':
    pythonfiles_location = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/ParamSearch/'
    results_folder_top = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Results/Aug24th/final_1/'
    IQfolder_high_level = '/Users/venkat/Documents/scisrs/Emanations/IQSamples/'
else:
    pythonfiles_location = '/home/vesathya/Emanations/Journal/Emanations/ParamSearch/'
    results_folder_top = '/home/vesathya/Emanations/Journal/PreliminaryPlots/Results/Aug9th/p1_p2search/'
    IQfolder_high_level = '/project/iarpa/Emanations/IQ/'
#
sys.path.insert(1, pythonfiles_location)
# sys.path.insert(1, './../../')
from EmanationDetection_search import *


########
######
def Iteration_perHyperParam(scenario, scenario_IQfolder_dict, CF_range, freq_slot_range, results_folder, \
                            results_folder_top, hyper_param_string, config_dict, use_trimmed):
    samprate = 200e6
    BW = 200  # Mhz
    results_dict = {}
    numtaps1 = 1000
    f_step1 = 25e6
    samprate_slice = f_step1
    kaiser_beta1 = 20
    trial_num = 0

    CF_slice_range = []
    #     for scenario in scenario_list:
    results_folder_scenario = results_folder + '/' + scenario + '/'
    duration_capture = scenario_IQfolder_dict[scenario]['durationcapture_ms']
    IQ_folder = IQfolder_high_level + scenario_IQfolder_dict[scenario]['IQFolder']

    plotval_dict = {}
    for CF in CF_range:
        for freq_slot in freq_slot_range:
            SF_freqslot, EF_freqslot = -samprate / 2 + (freq_slot) * f_step1, -samprate / 2 + (freq_slot + 1) * f_step1
            CF_freqslot = (SF_freqslot + EF_freqslot) / 2
            shift_freq = -1 * (SF_freqslot + EF_freqslot) / 2
            cutoff1 = f_step1 / 2
            ####### SLicing/bandpass filtering via kaiser filter

            #             print("Scenario is: ",scenario)
            #             print(" CF: ", CF_freqslot+CF*1e6)

            CF_slice_range.append(int((CF_freqslot + CF * 1e6) / 1e6))

    for CF_slice in CF_slice_range:  # np.arange(137, 1113, 25):
        if use_trimmed:
            results_file = results_folder + '/' + scenario + '/' + 'Scenario_' + scenario + '_CF_' + str(
                CF_slice) + 'MHz_trimmed.pkl'
        else:
            results_file = results_folder + '/' + scenario + '/' + 'Scenario_' + scenario + '_CF_' + str(
                CF_slice) + 'MHz.pkl'
        with open(results_file, 'rb') as file:
            resultval = pickle.load(file)
        file.close()
        plotval_dict[CF_slice] = {}
        for keyval in resultval['results'].keys():
            # We are picking the median SNR of the lowest 5 harmonics as the SNR for the entire harmonic series.
            SNR_val = np.median(resultval['results'][keyval]['SNR'][0:5])
            pitch_val = round(keyval, 2)
            #             print("CF: ", CF_slice, "MHz. Pitch: ", pitch_val, ". SNR: ", SNR_val)
            plotval_dict[CF_slice][pitch_val] = SNR_val
    #             sys.stdout = original_stdout
    Results_file = results_folder + '/' + scenario + '/' + 'plotval_dict.pkl'
    #     print(Results_file)
    with open(Results_file, 'wb') as f:
        pickle.dump(plotval_dict, f)


########################################################################################################################
# UPDATE YAML FILE
########################################################################################################################
# Function to update values in the YAML file
def update_yaml_file(hyper_param, file_path="synapse_emanation_search.yaml"):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Update the data
    #     yaml_data['EstimateHarmonic']['p_hh_1'] = float(str(hyper_param['p1']))
    #     yaml_data['EstimateHarmonic']['p_hh_2'] = float(str(hyper_param['p2']))
    #     yaml_data['EstimateHarmonic']['p_lh_1'] = float(str(hyper_param['p1']))
    #     yaml_data['EstimateHarmonic']['p_lh_2'] = float(str(hyper_param['p2']))
    return yaml_data


########################################################################################################################
########################################################################################################################

scenarios_complete_list = ['busy_conference_environ', 'Laptop_Nomonitor_NoUSBStick', 'Laptop_MonitorViaAdaptor', \
                           'Laptop_TwoMonitorsViaAdaptor', \
                           'Laptop_KeyboardDamaged', 'Laptop_MouseDamaged', 'Desktop_Monitor_CinebenchProcess',
                           'Laptop_SDCard_Datatransfer', \
                           'Laptop_ExternalHardDisk_Datatransfer', 'MonitortoLaptop_Bluetooth',
                           'SingleMonitorOnly_nolaptop', \
                           'TwoMonitors_nolaptop', 'LaptopOnly_nothingelse_June2ndCollect', 'Baseline_NoDesktop', \
                           'Test_withDesktop', 'Antennae3FeetAway', 'Antennae12FeetAway', \
                           'Mouse_Keyboard_Background', 'NoSource_June2nd', 'LaptopOnly_June2nd',
                           'LaptopConnectectedToAdaptorNoMonitor_June2nd', \
                           'LaptopNomonitor_USBStick_Kingston16GB', 'LaptopNoMonitor_USBStickSamsung']  # , 'no_source'

scenarios_final_list_plotting = ['Laptop_MonitorViaAdaptor', \
                                 'Laptop_KeyboardDamaged', 'Laptop_MouseDamaged', 'Desktop_Monitor_CinebenchProcess', \
                                 'Laptop_SDCard_Datatransfer', \
                                 'Laptop_ExternalHardDisk_Datatransfer',
                                 'LaptopOnly_June2nd', \
                                 'LaptopConnectectedToAdaptorNoMonitor_June2nd', \
                                 'LaptopNomonitor_USBStick_Kingston16GB', 'LaptopNoMonitor_USBStickSamsung']

keyboard_mouse_damaged = ['LaptopOnly_June2nd', 'Laptop_KeyboardDamaged', \
                          'Laptop_MouseDamaged']

external_storagedevice = ['LaptopOnly_June2nd', 'Laptop_SDCard_Datatransfer', \
                          'Laptop_ExternalHardDisk_Datatransfer', 'LaptopNomonitor_USBStick_Kingston16GB',
                          'LaptopNoMonitor_USBStickSamsung']
baseline_test_cage = ['Baseline_NoDesktop', 'Test_withDesktop', 'Desktop_Monitor_CinebenchProcess']
Laptop_usecases = ['LaptopOnly_June2nd', 'LaptopConnectectedToAdaptorNoMonitor_June2nd', 'Laptop_MonitorViaAdaptor']

external_storagedevice_final = external_storagedevice
Laptop_usecases_final = ['LaptopOnly_June2nd', 'LaptopConnectectedToAdaptorNoMonitor_June2nd',
                         'Laptop_MonitorViaAdaptor']
keyboard_mouse_damaged_final = ['LaptopOnly_June2nd', 'Laptop_KeyboardDamaged', 'Laptop_MouseDamaged']

scenario_IQfolder_dict = {}
for scenario_val in scenarios_complete_list:
    scenario_IQfolder_dict[scenario_val] = {}

scenario_IQfolder_dict['busy_conference_environ']['IQFolder'] = 'Nov8thCollect/Desktop2Monitor_CinebenchProgramRunning/'
scenario_IQfolder_dict['busy_conference_environ']['durationcapture_ms'] = '100'
# scenario_IQfolder_dict['no_source']['IQFolder'] = 'June2ndCollect/NoSource/'
# scenario_IQfolder_dict['no_source']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Laptop_Nomonitor_NoUSBStick'][
    'IQFolder'] = 'June16thCollect/LaptopNoMonitor_NoUSBStick_NoExternalStorage/'
scenario_IQfolder_dict['Laptop_Nomonitor_NoUSBStick']['durationcapture_ms'] = '300'
scenario_IQfolder_dict['Laptop_SDCard_Datatransfer']['IQFolder'] = 'Nov8thCollect/SdcardDatatransfer/'
scenario_IQfolder_dict['Laptop_SDCard_Datatransfer']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Laptop_ExternalHardDisk_Datatransfer'][
    'IQFolder'] = 'Nov8thCollect/ExternalHardDisk_Big_Datatransfer/'
scenario_IQfolder_dict['Laptop_ExternalHardDisk_Datatransfer']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Laptop_MonitorViaAdaptor']['IQFolder'] = 'June16thCollect/LaptopConnectedToMonitorviaAdaptor/'
scenario_IQfolder_dict['Laptop_MonitorViaAdaptor']['durationcapture_ms'] = '300'
scenario_IQfolder_dict['Laptop_TwoMonitorsViaAdaptor']['IQFolder'] = 'June16thCollect/Monitor1_2/'
scenario_IQfolder_dict['Laptop_TwoMonitorsViaAdaptor']['durationcapture_ms'] = '300'
scenario_IQfolder_dict['MonitortoLaptop_Bluetooth']['IQFolder'] = 'MonitortoLaptop_Bluetooth/'
scenario_IQfolder_dict['MonitortoLaptop_Bluetooth']['durationcapture_ms'] = '500'
scenario_IQfolder_dict['Desktop_Monitor_CinebenchProcess'][
    'IQFolder'] = 'Nov8thCollect/Desktop2Monitor_CinebenchProgramRunning/'
scenario_IQfolder_dict['Desktop_Monitor_CinebenchProcess']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Laptop_KeyboardDamaged']['IQFolder'] = 'Nov8thCollect/Keyboard_CableDamaged/'
scenario_IQfolder_dict['Laptop_KeyboardDamaged']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Laptop_MouseDamaged']['IQFolder'] = 'Nov8thCollect/Mouse_CableDamaged/'
scenario_IQfolder_dict['Laptop_MouseDamaged']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['SingleMonitorOnly_nolaptop']['IQFolder'] = 'June16thCollect/FirstMonitor/'
scenario_IQfolder_dict['SingleMonitorOnly_nolaptop']['durationcapture_ms'] = '300'
scenario_IQfolder_dict['TwoMonitors_nolaptop']['IQFolder'] = 'June16thCollect/Monitor1_2/'
scenario_IQfolder_dict['TwoMonitors_nolaptop']['durationcapture_ms'] = '300'
scenario_IQfolder_dict['LaptopOnly_nothingelse_June2ndCollect']['IQFolder'] = 'June2ndCollect/LaptopOnly/'
scenario_IQfolder_dict['LaptopOnly_nothingelse_June2ndCollect']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Baseline_NoDesktop']['IQFolder'] = 'Dec6thCollect/Baseline/'
scenario_IQfolder_dict['Baseline_NoDesktop']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Test_withDesktop']['IQFolder'] = 'Dec6thCollect/Test/'
scenario_IQfolder_dict['Test_withDesktop']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['Antennae3FeetAway']['IQFolder'] = 'Aug16thCollect/Antennae3FeetAway/'
scenario_IQfolder_dict['Antennae3FeetAway']['durationcapture_ms'] = '200'
scenario_IQfolder_dict['Antennae12FeetAway']['IQFolder'] = 'Aug16thCollect/Antennae12FeetAway/'
scenario_IQfolder_dict['Antennae12FeetAway']['durationcapture_ms'] = '200'
# 'Mouse_Keyboard_Background','NoSource_June2nd', 'LaptopOnly_June2nd','LaptopConnectectedToAdaptorNoMonitor_June2nd'
scenario_IQfolder_dict['Mouse_Keyboard_Background']['IQFolder'] = 'Nov8thCollect/Mouse_Keyboard_Background/'
scenario_IQfolder_dict['Mouse_Keyboard_Background']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['NoSource_June2nd']['IQFolder'] = 'June2ndCollect/NoSource/'
scenario_IQfolder_dict['NoSource_June2nd']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['LaptopOnly_June2nd']['IQFolder'] = 'June2ndCollect/LaptopOnly/'
scenario_IQfolder_dict['LaptopOnly_June2nd']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['LaptopConnectectedToAdaptorNoMonitor_June2nd'][
    'IQFolder'] = 'June2ndCollect/LaptopConnectectedToAdaptorNoMonitor/'
scenario_IQfolder_dict['LaptopConnectectedToAdaptorNoMonitor_June2nd']['durationcapture_ms'] = '100'
scenario_IQfolder_dict['LaptopNomonitor_USBStick_Kingston16GB'][
    'IQFolder'] = 'June16thCollect/LaptopNomonitor_USBStick_Kingston16GB/'
scenario_IQfolder_dict['LaptopNomonitor_USBStick_Kingston16GB']['durationcapture_ms'] = '300'
scenario_IQfolder_dict['LaptopNoMonitor_USBStickSamsung'][
    'IQFolder'] = 'June16thCollect/LaptopNoMonitor_USBStickSamsung/'
scenario_IQfolder_dict['LaptopNoMonitor_USBStickSamsung']['durationcapture_ms'] = '300'


########################################################################################################################
########################################################################################################################
# '/home/vesathya/Emanations/Journal/PreliminaryPlots/Results/' + \
#     'Aug8th/userinput/p1_0.30000000000000004_p2_0.30000000000000004_wt_100.0/' +
# '/home/vesathya/Emanations/Journal/PreliminaryPlots/Results/Aug8th/wtsearch/' + \
# 'p1_0.1_p2_0.3_wt_0.001/Laptop_Nomonitor_NoUSBStick/plotval_dict.pkl
# We noticed 1030 pitches cumulatively in those 8to 9 scenarios across 200 to 700 MHz.
def load_pitchsnr_values(scenario, hyper_param_string, results_folder_top):
    filename = results_folder_top + hyper_param_string + '/' + scenario + '/plotval_dict.pkl'
    with open(filename, 'rb') as f:
        plot_dict = pickle.load(f)
    return plot_dict


CF_range_start = 200
freq_slot_start = 0
CF_range_end = 1100  # 1100
freq_slot_end = 8
freq_slice = 25  # in MHz
# keyboard_mouse_damaged, external_storagedevice, baseline_test_cage, Laptop_usecases, ['Desktop_Monitor_CinebenchProcess']
scenario_list = scenarios_final_list_plotting  # ['Desktop_Monitor_CinebenchProcess']   #scenarios_complete_list
freq_slot_range = np.arange(freq_slot_start, freq_slot_end, 1)
CF_range = np.arange(CF_range_start, CF_range_end, 200)

hist2d_plot = True
hexbin_plot = False

use_trim = True
hyper_param = {}
# p1_range = [0.5]
# p2_range = [0.5]
# wt_range = np.array([0.001])
alternate_logscale = True  # we take log of values itself and not use log scale option for plotting.
# This is to debug a potential bug with matplotlib log2 scale plotting.
############################################ Plotting options #############################################
cmap = 'cool'  # cool, Wistia, grey, binary

y_bins = 50
log_base = 10

##########################################################################################################
# for hyper_param['p1'] in p1_range:
#     for hyper_param['p2'] in p2_range:
s_range = [-1]
for hyper_param['s1'] in s_range:
    for hyper_param['s2'] in s_range:
        for scenario in scenario_list:
            print(scenario)
            if scenario in ['Laptop_MonitorViaAdaptor', 'Desktop_Monitor_CinebenchProcess']:
                if alternate_logscale == False:
                    ylim_start = 1e1
                    ylim_end = 3e6
                else:
                    ylim_start = np.emath.logn(log_base, 1e1)
                    ylim_end = np.emath.logn(log_base, 2e6)
            else:
                if alternate_logscale == False:
                    ylim_start = 5e4
                    ylim_end = 3e6
                else:
                    ylim_start = np.emath.logn(log_base, 9e4)
                    ylim_end = np.emath.logn(log_base, 2e6)

            hyper_param_string = 's1_' + str(hyper_param['s1']) + 's2_' + str(hyper_param['s2'])
            config_dict = update_yaml_file(hyper_param)
            results_folder = results_folder_top + hyper_param_string
            #                 hyper_param_string = 'p1_'+str(hyper_param['p1']) + '_p2_'+str(hyper_param['p2'])
            Iteration_perHyperParam(scenario, scenario_IQfolder_dict, CF_range, freq_slot_range, results_folder, \
                                    results_folder_top, hyper_param_string, config_dict, use_trim)
            plot_dict = load_pitchsnr_values(scenario, hyper_param_string, results_folder_top)

            from matplotlib import colors

            y, x, weights = [], [], []
            for CF_key in plot_dict.keys():
                for pitch_key in plot_dict[CF_key].keys():
                    SNR = plot_dict[CF_key][pitch_key]
                    x.append(CF_key)
                    y.append(pitch_key)
                    weights.append(SNR)
            if alternate_logscale == True:
                y = np.emath.logn(log_base, y)
            CF_slice_start = np.min(list(plot_dict.keys())) - freq_slice / 2
            CF_slice_end = np.max(list(plot_dict.keys())) + freq_slice / 2
            x_bins = np.arange(CF_slice_start, CF_slice_end, freq_slice)

            x_bins_hist2d = x_bins

            if alternate_logscale == False:
                y_bins_hist2d = np.logspace(np.emath.logn(log_base, ylim_start), \
                                            np.emath.logn(log_base, ylim_end), y_bins)
            else:
                y_bins_hist2d = np.linspace(ylim_start, ylim_end, num=y_bins)

            #                 x_gridsize = int((CF_slice_end - CF_slice_start)/25)
            #                 y_gridsize = 50
            ylim = [ylim_start, ylim_end]
            xlim = [CF_slice_start, CF_slice_end]
            # H, yedges, xedges = np.histogram2d(x, y, bins = [x_bins, y_bins], weights = weights)
            #                 print(y)
            #                 if hexbin_plot == True:
            #                     fig, ax = plt.subplots(tight_layout=True)
            #                     # hist = ax.hist2d(y, x, bins = [y_bins, x_bins], weights = weights, norm=colors.LogNorm())

            #                     # H, xedges, yedges = np.histogram2d(x, y, C = weights, extent = [0,5000,CF_slice_start, CF_slice_end], \
            #                     #                     yscale = 'log', gridsize=(200,y_gridsize))

            #                     hb = ax.hexbin(x,y, C = weights, yscale = 'log',\
            #                                         gridsize=(x_gridsize,y_gridsize), cmap='viridis', alpha=0.9)
            #                     ax.set(ylim=ylim, xlim=xlim)
            #                     ax.set_title("hexbin_plot" + " and " + scenario)
            #                     cb = fig.colorbar(hb, ax=ax)
            #                     plt.gcf().set_size_inches(8, 4)
            #                     plt.show()
            #                     import time
            # #                     time.sleep(5)

            if hist2d_plot == True:
                import matplotlib.colors as mcolors

                #                     fig, ax = plt.subplots(tight_layout=True)
                #plt.figure()

                #                     plt.title("Pitch heatmap " + scenario, fontsize = 30)# + ' scenario_'+scenario

                # (log' + str(log_base) + ' scale).'
                #                     H, xedges, yedges = np.histogram2d(y,x, weights = weights, bins = (y_bins_hist2d, x_bins_hist2d))
                #                     im = ax.imshow(H.T, cmap='inferno', interpolation='None', norm=matplotlib.colors.LogNorm())

                #                     ax.set(ylim=ylim, xlim=xlim)
                #                     ax.set_title("his2d_plot" + " and " + scenario)
                #                     ax.set_xlabel('IQ capture freq. in MHz.')
                #                     ax.set_ylabel('Pitch freq. in Hz (log' + str(log_base) + ' scale).')
                #                     plt.gcf().set_size_inches(10, 4)
                #                     ax.set_xscale('linear')



                #                         ax.set_yscale('log', basey=log_base)
                #                     plt.yscale('log')
                #                     ax.set_yscale('log', basey=log_base)
                #                     ax.set_yscale('log', base=log_base)
                #                     ax.set_yticks()
                #                     plt.show()
                #                     plt.title("Power spectral density", fontsize=12)
                #                     plt.xlabel(x_label, fontsize=10)
                #                     plt.ylabel("Power (dB)", fontsize=10)

                #                     plt.yticks(np.arange(1,7,1), [r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$'],\
                #                                fontsize=14)


                cmin_plot1 = 0.01
                cmin_plot2 = 0.01
                cmax_plot1 = 35
                cmax_plot2 = 25
                if scenario in ['Laptop_MonitorViaAdaptor', 'Desktop_Monitor_CinebenchProcess']:
                    plt.hist2d(x, y, weights=weights, bins=[x_bins_hist2d, y_bins_hist2d], \
                               cmap=cmap, cmin=cmin_plot1, cmax=cmax_plot1,
                               norm=plt.Normalize(vmin=cmin_plot1, vmax=cmax_plot1))  # cool, Wistia, grey, binary
                    plt.ylim(ylim)
                    if alternate_logscale == False:
                        plt.yscale('log', basey=log_base)
                    scaling = 2
                    cbar_label_fontsize = scaling * 20
                    cbar_tick_fontsize = scaling * 20
                    xtick_fontsize = scaling * 20
                    ytick_fontsize = scaling * 20
                    xlabel_fontsize = scaling * 20
                    ylabel_fontsize = scaling * 20
                    plt.yticks(np.arange(1, 7, 1), [r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'],
                               fontsize=ytick_fontsize)
                    plt.xticks(fontsize=xtick_fontsize)
                    if scenario in ['Laptop_MonitorViaAdaptor']:
                        plt.xlabel('IQ capture frequency (MHz)', fontsize=xlabel_fontsize)
                        plt.ylabel('Pitch frequency (Hz)', fontsize=ylabel_fontsize)
                    else:
                        plt.xlabel('IQ capture frequency (MHz)', fontsize=xlabel_fontsize)
                        cbar = plt.colorbar(ticks=np.arange(0,cmax_plot1+5, 5))  # hb[3], ax=ax
                        # cbar.ax.set_yticks()
                        # ticks=[0, 5, 10, 15, 20, 25, 30, 35]
                        #                     cbar.set_ticks([0,5,10,15,20,25,30,35])
                        #                     cbar.set_ticklabels(['0','5','10','15','20','25','30','35'])
                        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
                        #                     cbar.ax.get_yaxis().set_ticks([0,5,10,15,20,25,30,35])
                        #                     for j, lab in enumerate(['$0$','$1$','$2$','$>3$']):
                        #                         cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
                        #                     cbar.ax.get_yaxis().labelpad = 15
                        cbar.ax.set_ylabel('SNR (dB)', fontsize=cbar_label_fontsize)
                    # ['Laptop_MonitorViaAdaptor', \
                    #  'Laptop_KeyboardDamaged', 'Laptop_MouseDamaged', 'Desktop_Monitor_CinebenchProcess', \
                    #  'Laptop_SDCard_Datatransfer', \
                    #  'Laptop_ExternalHardDisk_Datatransfer',
                    #  'LaptopOnly_June2nd', \
                    #  'LaptopConnectectedToAdaptorNoMonitor_June2nd', \
                    #  'LaptopNomonitor_USBStick_Kingston16GB', 'LaptopNoMonitor_USBStickSamsung']
                else:
                    plt.hist2d(x, y, weights=weights, bins=[x_bins_hist2d, y_bins_hist2d], \
                               cmap=cmap, cmin=cmin_plot2, cmax=cmax_plot2,
                               norm=plt.Normalize(vmin=cmin_plot2, vmax=cmax_plot2))  # cool, Wistia, grey, binary
                    plt.ylim(ylim)
                    if alternate_logscale == False:
                        plt.yscale('log', basey=log_base)
                    scaling = 2.2
                    cbar_label_fontsize = scaling * 20
                    cbar_tick_fontsize = scaling * 20
                    xtick_fontsize = scaling * 20
                    ytick_fontsize = scaling * 20
                    xlabel_fontsize = scaling * 25
                    ylabel_fontsize = scaling * 25
                    if scenario in ['LaptopOnly_June2nd']:
                        plt.yticks(np.arange(5, 7, 1), [r'$10^5$', r'$10^6$'], fontsize=ytick_fontsize)
                        plt.xticks(fontsize=xtick_fontsize)
                        plt.ylabel('Pitch frequency (Hz)', fontsize=ylabel_fontsize)
                        plt.xlabel('IQ capture frequency (MHz)', fontsize=xlabel_fontsize)
                    elif scenario in ['Laptop_KeyboardDamaged', 'LaptopNoMonitor_USBStickSamsung']:
                        plt.yticks([])#(np.arange(5, 7, 1), [r'$10^5$', r'$10^6$'], fontsize=ytick_fontsize)
                        plt.xticks(fontsize=xtick_fontsize)
                        plt.xlabel('IQ capture frequency (MHz)', fontsize=xlabel_fontsize)
                        # plt.ylabel('Pitch frequency (Hz)', fontsize=ylabel_fontsize)
                        cbar = plt.colorbar(ticks=np.arange(0,cmax_plot2+5, 5))  # hb[3], ax=ax
                        # cbar.ax.set_yticks()
                        # ticks=[0, 5, 10, 15, 20, 25, 30, 35]
                        #                     cbar.set_ticks([0,5,10,15,20,25,30,35])
                        #                     cbar.set_ticklabels(['0','5','10','15','20','25','30','35'])
                        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
                        #                     cbar.ax.get_yaxis().set_ticks([0,5,10,15,20,25,30,35])
                        #                     for j, lab in enumerate(['$0$','$1$','$2$','$>3$']):
                        #                         cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
                        #                     cbar.ax.get_yaxis().labelpad = 15
                        cbar.ax.set_ylabel('SNR (dB)', fontsize=cbar_label_fontsize)
                    else:
                        plt.yticks([])#(np.arange(5, 7, 1), [r'$10^5$', r'$10^6$'], fontsize=ytick_fontsize)
                        plt.xticks(fontsize=xtick_fontsize)
                        plt.xlabel('IQ capture frequency (MHz)', fontsize=xlabel_fontsize)
                        # xtick_fontsize = 14
                        # cbar_fontsize = 20
                        # xlabel_fontsize = 20
                        # ylabel_fontsize = 20



                #                     plt.legend()
                # fig.tight_layout()
                plt.gcf().set_size_inches(15, 10)
                plt.subplots_adjust(left=0.12,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.95,
                                    wspace=0.25,
                                    hspace=0.25)
                plt.tight_layout()
                results_folder_scenario = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/StandaloneRunningCode/Results/'
                # '/Users/venkat/Desktop/PlotFiles/'#results_folder +'/' + scenario + '/'
                plt.savefig(results_folder_scenario + 'Heatmap_' + scenario + '.pdf', \
                            format='pdf', bbox_inches='tight', pad_inches=.01)
                plt.clf()
                # plt.show()
                # plt.close()
