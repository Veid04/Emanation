import numpy as np
import yaml
import copy
import math
#/home/vesathya/Emanations/Journal/Emanations_JournalCode/Emanations/synapse_emanation.yaml
# def config_EstimateHarmonics(yaml_filename ="../synapse_emanation.yaml"):
#     with open(yaml_filename, 'r') as config_file:
#         config = yaml.safe_load(config_file)
# #     global q1, q2, r1, r2, s1, s2, num_steps_coarse, fss1, fss2, fss3, lowest_fh, fse1, fse2, fine_search_length, num_steps_finesearch, \
# #         Err_thresh_dict, num_meas1, num_meas2, n1_fh, n2_fh
#     ########################################################################################################################
#     ########################################################################################################################
#     # Config file choices
#     # Freq. estimate objective function hyper-parameters
#     q1 = config['EstimateHarmonic']['q1']
#     q2 = config['EstimateHarmonic']['q2']
#     r1 = config['EstimateHarmonic']['r1']
#     r2 = config['EstimateHarmonic']['r2']
#     s1 = config['EstimateHarmonic']['s1']
#     s2 = config['EstimateHarmonic']['s2']
#     p_hh_1 = config['EstimateHarmonic']['p_hh_1']
#     p_lh_1 = config['EstimateHarmonic']['p_lh_1']
#     p_hh_2 = config['EstimateHarmonic']['p_hh_2']
#     p_lh_2 = config['EstimateHarmonic']['p_lh_2']
#     wt_meas_pred_hh = config['EstimateHarmonic']['wt_meas_pred_hh']
#     wt_meas_pred_lh = config['EstimateHarmonic']['wt_meas_pred_lh']

#     num_steps_coarse = config['EstimateHarmonic']['num_steps_coarse'] #
#     # freq. search grid points start and end values.
#     fss1 = config['EstimateHarmonic']['fss1']
#     fss2 = config['EstimateHarmonic']['fss2']
#     fss3 = config['EstimateHarmonic']['fss3']
#     lowest_fh = config['EstimateHarmonic']['lowest_fh']
#     highest_fh_withoutInc = config['EstimateHarmonic']['highest_fh_withoutInc']
#     fse1 = config['EstimateHarmonic']['fse1']
#     fse2 = config['EstimateHarmonic']['fse2']

#     # Fine search lengths
#     fine_search_length = config['EstimateHarmonic']['fine_search_length']  # we search within 4 indices frequency range - around the index identifed as the minima in coarse search
#     # We can change this value? This can help identify 222 as peak over 111 ?
#     num_steps_finesearch = config['EstimateHarmonic']['num_steps_finesearch']  # Number of search points in fine search

#     # Dictionary of keys that are frequency limits, under which and closest if we find a fund_harmonic, we return a corresponding
#     # percentage value for frequency error thresholding. This dictionary can grow in size.
#     # 100e9 is picked as a very high limit such as Inf
#     Err_thresh_dict = config['EstimateHarmonic']['Err_thresh_dict']

#     # We need atleast num_meas1 measured peaks whose freq. is under n1_fh times fh, and atleast num_meas2 under n2_fh times fh
#     num_meas1 = config['EstimateHarmonic']['num_meas1']
#     num_meas2 = config['EstimateHarmonic']['num_meas2']
#     n1_fh = config['EstimateHarmonic']['n1_fh']
#     n2_fh = config['EstimateHarmonic']['n2_fh']
#     non_vectorized = config['EstimateHarmonic']['non_vectorized']
#     fine_search_fundharmonic_estimate = config['EstimateHarmonic']['fine_search_fundharmonic_estimate']
#     # num_runs_fundharmonic_estimate_perfreqslot = config['EstimateHarmonic']['num_runs_fundharmonic_estimate_perfreqslot']
    
#     return p_lh_1, p_lh_2, p_hh_1, p_hh_2, q1, q2, r1, r2, s1, s2, wt_meas_pred_lh, wt_meas_pred_hh, num_steps_coarse, fss1, fss2, fss3, \
#            lowest_fh, highest_fh_withoutInc, fse1, fse2, fine_search_length, num_steps_finesearch, \
#         Err_thresh_dict, num_meas1, num_meas2, n1_fh, n2_fh, non_vectorized, fine_search_fundharmonic_estimate
########################################################################################################################
########################################################################################################################

# def error_bound_freq_perc(fh):
#     if harmonic_n < 200:
#         error_bound_freq_perc = 3
#     elif harmonic_n < 1000:
#         error_bound_freq_perc = 2
#     else:
#         error_bound_freq_perc = error_bound_freq_perc_fromuser


# p_lh_1, p_lh_2, p_hh_1, p_hh_2, q1, q2, r1, r2, s1, s2, wt_meas_pred_lh, wt_meas_pred_hh, num_steps_coarse, fss1, fss2, fss3, lowest_fh,\
#  highest_fh_withoutInc, fse1, fse2, fine_search_length, num_steps_finesearch, \
#         Err_thresh_dict, num_meas1, num_meas2, n1_fh, n2_fh, non_vectorized, fine_search_fundharmonic_estimate = config_dict['EstimateHarmonic']['p_lh_1'], config_dict['EstimateHarmonic']['p_lh_2'], config_dict['EstimateHarmonic']['p_hh_1'], config_dict['EstimateHarmonic']['p_hh_2'], config_dict['EstimateHarmonic']['q1'], config_dict['EstimateHarmonic']['q2'], config_dict['EstimateHarmonic']['r1'], config_dict['EstimateHarmonic']['r2'], config_dict['EstimateHarmonic']['s1'], config_dict['EstimateHarmonic']['s2'], config_dict['EstimateHarmonic']['wt_meas_pred_lh'], config_dict['EstimateHarmonic']['wt_meas_pred_hh'], config_dict['EstimateHarmonic']['num_steps_coarse'], config_dict['EstimateHarmonic']['fss1'], config_dict['EstimateHarmonic']['fss2'], config_dict['EstimateHarmonic']['fss3'], config_dict['EstimateHarmonic']['lowest_fh'],\
#  config_dict['EstimateHarmonic']['highest_fh_withoutInc'], config_dict['EstimateHarmonic']['fse1'], config_dict['EstimateHarmonic']['fse2'], config_dict['EstimateHarmonic']['fine_search_length'], config_dict['EstimateHarmonic']['num_steps_finesearch'], \
#         config_dict['EstimateHarmonic']['Err_thresh_dict'], config_dict['EstimateHarmonic']['num_meas1'], config_dict['EstimateHarmonic']['num_meas2'], config_dict['EstimateHarmonic']['n1_fh'], config_dict['EstimateHarmonic']['n2_fh'], config_dict['EstimateHarmonic']['non_vectorized'], config_dict['EstimateHarmonic']['fine_search_fundharmonic_estimate']

# Dictionary of keys that are frequency limits, under which and closest if we find a fund_harmonic, we return a corresponding
# percentage value for frequency error thresholding.
def error_threshold_harmonic(fund_harmonic, config_dict):
    p_lh_1, p_lh_2, p_hh_1, p_hh_2, q1, q2, r1, r2, s1, s2, wt_meas_pred_lh, wt_meas_pred_hh, num_steps_coarse, fss1, fss2, fss3, lowest_fh,\
 highest_fh_withoutInc, fse1, fse2, fine_search_length, num_steps_finesearch, \
        Err_thresh_dict, num_meas1, num_meas2, n1_fh, n2_fh, non_vectorized, fine_search_fundharmonic_estimate = config_dict['EstimateHarmonic']['p_lh_1'], config_dict['EstimateHarmonic']['p_lh_2'], config_dict['EstimateHarmonic']['p_hh_1'], config_dict['EstimateHarmonic']['p_hh_2'], config_dict['EstimateHarmonic']['q1'], config_dict['EstimateHarmonic']['q2'], config_dict['EstimateHarmonic']['r1'], config_dict['EstimateHarmonic']['r2'], config_dict['EstimateHarmonic']['s1'], config_dict['EstimateHarmonic']['s2'], config_dict['EstimateHarmonic']['wt_meas_pred_lh'], config_dict['EstimateHarmonic']['wt_meas_pred_hh'], config_dict['EstimateHarmonic']['num_steps_coarse'], config_dict['EstimateHarmonic']['fss1'], config_dict['EstimateHarmonic']['fss2'], config_dict['EstimateHarmonic']['fss3'], config_dict['EstimateHarmonic']['lowest_fh'],\
 config_dict['EstimateHarmonic']['highest_fh_withoutInc'], config_dict['EstimateHarmonic']['fse1'], config_dict['EstimateHarmonic']['fse2'], config_dict['EstimateHarmonic']['fine_search_length'], config_dict['EstimateHarmonic']['num_steps_finesearch'], \
        config_dict['EstimateHarmonic']['Err_thresh_dict'], config_dict['EstimateHarmonic']['num_meas1'], config_dict['EstimateHarmonic']['num_meas2'], config_dict['EstimateHarmonic']['n1_fh'], config_dict['EstimateHarmonic']['n2_fh'], config_dict['EstimateHarmonic']['non_vectorized'], config_dict['EstimateHarmonic']['fine_search_fundharmonic_estimate']
    
    err_thresh_freq = np.sort(np.array(list(Err_thresh_dict.keys()))) #np.sort(np.array(Err_thresh_dict.keys()))

    idx_limit = np.min(np.argwhere((err_thresh_freq - fund_harmonic) > 0))
    perc_error = Err_thresh_dict[err_thresh_freq[idx_limit]]
    return perc_error

# return numbers within a range from a list. This is used for fine freq search. We prune measured partials in fine search,
# to be close to the potential candidates of pitch.



def Error_ObjectiveFunctionEstimate(a_est, f_est,wt_meas_pred,freq_search,p1,p2, config_dict):
    p_lh_1, p_lh_2, p_hh_1, p_hh_2, q1, q2, r1, r2, s1, s2, wt_meas_pred_lh, wt_meas_pred_hh, num_steps_coarse, fss1, fss2, fss3, lowest_fh,\
 highest_fh_withoutInc, fse1, fse2, fine_search_length, num_steps_finesearch, \
        Err_thresh_dict, num_meas1, num_meas2, n1_fh, n2_fh, non_vectorized, fine_search_fundharmonic_estimate = config_dict['EstimateHarmonic']['p_lh_1'], config_dict['EstimateHarmonic']['p_lh_2'], config_dict['EstimateHarmonic']['p_hh_1'], config_dict['EstimateHarmonic']['p_hh_2'], config_dict['EstimateHarmonic']['q1'], config_dict['EstimateHarmonic']['q2'], config_dict['EstimateHarmonic']['r1'], config_dict['EstimateHarmonic']['r2'], config_dict['EstimateHarmonic']['s1'], config_dict['EstimateHarmonic']['s2'], config_dict['EstimateHarmonic']['wt_meas_pred_lh'], config_dict['EstimateHarmonic']['wt_meas_pred_hh'], config_dict['EstimateHarmonic']['num_steps_coarse'], config_dict['EstimateHarmonic']['fss1'], config_dict['EstimateHarmonic']['fss2'], config_dict['EstimateHarmonic']['fss3'], config_dict['EstimateHarmonic']['lowest_fh'],\
 config_dict['EstimateHarmonic']['highest_fh_withoutInc'], config_dict['EstimateHarmonic']['fse1'], config_dict['EstimateHarmonic']['fse2'], config_dict['EstimateHarmonic']['fine_search_length'], config_dict['EstimateHarmonic']['num_steps_finesearch'], \
        config_dict['EstimateHarmonic']['Err_thresh_dict'], config_dict['EstimateHarmonic']['num_meas1'], config_dict['EstimateHarmonic']['num_meas2'], config_dict['EstimateHarmonic']['n1_fh'], config_dict['EstimateHarmonic']['n2_fh'], config_dict['EstimateHarmonic']['non_vectorized'], config_dict['EstimateHarmonic']['fine_search_fundharmonic_estimate']
    
    
    
    # We are doing flatten in place. To avoid any issues of doing this in main copy, we process over a deepcopy
    a_est_copy = copy.deepcopy(a_est)
    f_est_copy = copy.deepcopy(f_est)
    K = len(a_est_copy)
    f_max, f_min, A_max = np.amax(f_est_copy), np.min(f_est_copy), np.max(a_est_copy)
    # A_max = 1
    R = len(freq_search)

    # wt_meas_pred = 200
    # p1=p2=p

    N = 1 #assigning dummy value, incase we do not enter the loop? error handling?
    # non_vectorized = True
    # print("s1 and s2 are: ",s1, s2)
    if non_vectorized:
        Err_pred_meas = np.zeros(R)
        Err_meas_pred = np.zeros(R)
        for r in range(R):
            f_fund = freq_search[r]
            N = np.ceil(f_max/f_fund).astype(int)
            a_n = np.zeros(N)
            f_n = np.zeros(N)
            d_fn = np.zeros(N)

            for n in range(N):
                f_n[n] = (n+1)*f_fund
                d_fn[n] = np.min(np.abs(f_n[n] - f_est_copy))
                idx = np.argmin(np.abs(f_n[n] - f_est_copy))
                a_n[n] = a_est_copy[idx]
                Err_pred_meas[r] += d_fn[n]/(f_n[n]**p1) + ((a_n[n]/A_max)**s1)*(((q1*d_fn[n])/((f_n[n])**p1)) - r1)

            a_k = np.zeros(K)
            d_fk = np.zeros(K)
            for k in range(K):
                d_fk[k] = np.min(np.abs(f_n - f_est_copy[k]))
                idx2 = np.argmin(np.abs(f_n - f_est_copy[k]))
                a_k[k] = a_n[idx2]
                # f_k[k] = f_n[idx]
                Err_meas_pred[r] += (d_fk[k]/(f_est_copy[k]**p2)) + \
                                    ((a_k[k]/A_max)**s2)*(((q2*d_fk[k])/(f_est_copy[k]**p2)) - r2)


        Err_total = Err_pred_meas/N +wt_meas_pred*Err_meas_pred/K
        return Err_total
    else:
        N_range = np.ceil(f_max / freq_search).astype(int)
        Err_pred_meas_arr = np.zeros((R,))
        Err_meas_pred_arr = np.zeros((R,))
        # a_est_copy.flatten()
        # f_est_copy.flatten()
        if len(f_est_copy.shape)==1:
            f_est_copy = np.expand_dims(f_est_copy,axis=1)

        if len(a_est_copy.shape)==1:
            a_est_copy = np.expand_dims(a_est_copy,axis=1)

        for rr in range(R):
            # a_n = np.zeros(N)
            # f_n = np.zeros(N)
            # d_fn = np.zeros(N)
            f_tile = np.tile(f_est_copy, N_range[rr])
            f_n1 = freq_search[rr] * (np.arange(N_range[rr]) + 1)
            d_fn1 = np.min(abs(f_n1 - f_tile), axis=0)
            idx_n = np.argmin(abs(f_n1 - f_tile), axis=0)
            a_n1 = a_est_copy[idx_n]
            fn1_p1_inv= 1/f_n1**p1
            Err_pred_meas_arr[rr] = np.sum(d_fn1 * (fn1_p1_inv) + ((a_n1.flatten() / A_max)** s1) * (
                    ((q1 * d_fn1) * (fn1_p1_inv)) - r1))

            fn_tile = np.tile(f_n1, K)
            d_fk1 = np.min(abs(f_est_copy - fn_tile), axis=1)
            idx2_n = np.argmin(abs(f_est_copy - fn_tile), axis=1)
            a_k1 = a_n1[idx2_n]
            # o1 = (d_fk1 / (f_est_copy ** p2))
            # o2 = ((a_k1.flatten() / A_max) ** s2)
            # o3 = (((q2 * d_fk1) / (f_est_copy ** p2)) - r2)
            # o4 = o2*o3
            # o5 = o1+o4
            fest_p2_inv = 1/f_est_copy.flatten()** p2
            Err_meas_pred_arr[rr] = np.sum((d_fk1 * (fest_p2_inv)) + ((a_k1.flatten() / A_max)** s2) * (
                        ((q2 * d_fk1) * (fest_p2_inv)) - r2))

            # ac = np.sum(Err_meas_pred_arr)
            # print(ac)

        # Err_total_arr = Err_pred_meas_arr / N_range[rr] + wt_meas_pred * Err_meas_pred_arr / K
        Err_total_arr = np.divide(np.reshape(Err_pred_meas_arr, N_range.shape), N_range)+ \
                        wt_meas_pred * Err_meas_pred_arr / K
        return Err_total_arr


def EstimateHarmonic(a_est, f_est, numpeaks_crossthresh, high_ff_search,config_dict):
    p_lh_1, p_lh_2, p_hh_1, p_hh_2, q1, q2, r1, r2, s1, s2, wt_meas_pred_lh, wt_meas_pred_hh, num_steps_coarse, fss1, fss2, fss3, lowest_fh,\
 highest_fh_withoutInc, fse1, fse2, fine_search_length, num_steps_finesearch, \
        Err_thresh_dict, num_meas1, num_meas2, n1_fh, n2_fh, non_vectorized, fine_search_fundharmonic_estimate = config_dict['EstimateHarmonic']['p_lh_1'], config_dict['EstimateHarmonic']['p_lh_2'], config_dict['EstimateHarmonic']['p_hh_1'], config_dict['EstimateHarmonic']['p_hh_2'], config_dict['EstimateHarmonic']['q1'], config_dict['EstimateHarmonic']['q2'], config_dict['EstimateHarmonic']['r1'], config_dict['EstimateHarmonic']['r2'], config_dict['EstimateHarmonic']['s1'], config_dict['EstimateHarmonic']['s2'], config_dict['EstimateHarmonic']['wt_meas_pred_lh'], config_dict['EstimateHarmonic']['wt_meas_pred_hh'], config_dict['EstimateHarmonic']['num_steps_coarse'], config_dict['EstimateHarmonic']['fss1'], config_dict['EstimateHarmonic']['fss2'], config_dict['EstimateHarmonic']['fss3'], config_dict['EstimateHarmonic']['lowest_fh'],\
 config_dict['EstimateHarmonic']['highest_fh_withoutInc'], config_dict['EstimateHarmonic']['fse1'], config_dict['EstimateHarmonic']['fse2'], config_dict['EstimateHarmonic']['fine_search_length'], config_dict['EstimateHarmonic']['num_steps_finesearch'], \
        config_dict['EstimateHarmonic']['Err_thresh_dict'], config_dict['EstimateHarmonic']['num_meas1'], config_dict['EstimateHarmonic']['num_meas2'], config_dict['EstimateHarmonic']['n1_fh'], config_dict['EstimateHarmonic']['n2_fh'], config_dict['EstimateHarmonic']['non_vectorized'], config_dict['EstimateHarmonic']['fine_search_fundharmonic_estimate']
    
    
    
    
    # freq_res = 6*Fb
    K = len(a_est)
    f_max, f_min, A_max = np.amax(f_est), np.min(f_est), np.max(a_est)
    # f_step = np.min([Fb/10, f_min/10])
    # freq_search = np.arange(f_min/4, np.percentile(f_est, 60)+f_step, f_step)

    # we search for thousand points int he search space.
    # search space is a robust number. We have had issues with starting peak very close to zero - noise blip.
    # This causes a huge search space. We thus use a percentile approach, but fix number of points in search space
    # If there are say 50 peaks, we take the frequency of the 25th peak and divide it by the number of peaks (50).
    # this is the start of search space. End of search space (possible candidate for fundamental harmonic) is chosen as 80 percentile.
    # We are expecting not to have more than 50 percentile of erroneous peaks.
    # these are possible candidates for fundamental harmonic

    # freq_search_start = np.min([np.percentile(f_est,50)/5*len(f_est), np.percentile(f_est,5)])
    # We sometimes have a noisy peak that is very close to zero, not part of the harmonic series.
    # this makes the search space very large, consuming enormous time.
    # we therefore take a robust frequency search space start estimate.
    # explanation for lowest_fh Hz is given in comments below in line 85. We do not want to detect any harmonic
    # that is lower than lowest_fh Hz

    # freq_search_start = np.max([np.percentile(f_est, fss1) / (fss2 * len(f_est)), np.min(f_est) / fss3, lowest_fh])
    # what if percentile 50 belongs to different harmonic series and percentile 80 to different.
    # freq_search_end = np.min([np.percentile(f_est, fse1) * (len(f_est)), np.percentile(f_est, fse2)])

    # Added new frequncy search


    if high_ff_search:
        freq_search_start = np.max([lowest_fh, np.percentile(f_est, 20) / (10 * len(f_est))])
        freq_search_end = np.percentile(f_est, 20)
        freq_search = np.linspace(freq_search_start, freq_search_end, num_steps_coarse)
        p1 = p_hh_1
        p2 = p_hh_2
        wt = wt_meas_pred_hh
    else:
        freq_search = np.arange(lowest_fh, highest_fh_withoutInc, 1)
        p1 = p_lh_1
        p2 = p_lh_2
        wt = wt_meas_pred_lh

    # p_coarse = p
    Err_total = Error_ObjectiveFunctionEstimate(a_est, f_est,wt,freq_search, p1, p2, config_dict)
    min_idx = np.argmin(Err_total)
    fund_harmonic_freq_coarse = freq_search[min_idx]
    Err_total_fine = []
    freq_search_fine = []
    # We do fine search only for high fundamental frequency search. For low harmoincs search, we have an exhaustive search space already in coarse search.
    if high_ff_search:
        if fine_search_fundharmonic_estimate:
            # Fine frequency search

            # start_idx_fine = int(np.max([0, min_idx - fine_search_length/2]))
            # end_idx_fine = int(np.min([len(Err_total)-1, min_idx + fine_search_length/2]))
            #
            # freq_search_fine = np.linspace(freq_search[start_idx_fine], freq_search[end_idx_fine], num_steps_finesearch)
            freq_search_fine = np.linspace(0.8*fund_harmonic_freq_coarse, \
                                           1.2*fund_harmonic_freq_coarse,num_steps_finesearch)
            # p_fine = p
            # wt_meas_pred_fine = 1000

            # We prune measured partials in fine search,
            # to be close to the potential candidates of pitch.
            start_range_meas_partial = np.min(freq_search_fine)
            end_range_meas_partial = 100*np.max(freq_search_fine)
            f_pruned = []
            a_pruned = []
            for idx_prune in range(len(f_est)):
                f_val = f_est[idx_prune]
                a_val = a_est[idx_prune]
                if start_range_meas_partial <= f_val <= end_range_meas_partial:
                    f_pruned.append(f_val.tolist())
                    a_pruned.append(a_val.tolist())
            if len(f_pruned) <=1:
                return [], [], [], []


            f_pruned = np.array(f_pruned)
            a_pruned = np.array(a_pruned)

            Err_total_fine = Error_ObjectiveFunctionEstimate(a_pruned, f_pruned, wt, freq_search_fine, p1, p2, config_dict)
            min_idx_fine = np.argmin(Err_total_fine)
            fund_harmonic_freq = freq_search_fine[min_idx_fine]
        else:
            fund_harmonic_freq = freq_search[min_idx]
    else:
        fund_harmonic_freq = fund_harmonic_freq_coarse




    harmonic_freq = []
    harmonic_SNR = []
    harmonic_idx = []
    # sometimes if we wrongly detect very low fundamental frequencies, we might see
    # multiples of it, with an error bound we specify. To avoid this, we have this check.
    # Typically the lowest we see is 50 Hz from monitor. Therefore this is a good number.
    if fund_harmonic_freq >= lowest_fh:

    # error_threshold_harmonic = 1
        N = np.ceil(f_max/fund_harmonic_freq).astype(int)
        for i in np.arange(1,N+1):
            harmonic_n = i*fund_harmonic_freq
            M_idx = np.argmin(np.abs(f_est - harmonic_n))
            f_closest = f_est[M_idx]
            err_thresh = error_threshold_harmonic(f_closest, config_dict)
            if np.abs(f_closest - harmonic_n) <= ((err_thresh/100)*fund_harmonic_freq):
                if f_closest not in harmonic_freq:
                    harmonic_freq.append(f_closest)
                    harmonic_SNR.append(a_est[M_idx])
        # Atleast three measurables should be within 10th estimated harmonic and
        # 5 measurables within 20th estimated harmonic
        if len(harmonic_freq)>=numpeaks_crossthresh:

            if n1_fh*fund_harmonic_freq > np.sort(harmonic_freq)[num_meas1-1]:
                   # and n2_fh*fund_harmonic_freq > np.sort(harmonic_freq)[num_meas2-1]:
                # return fund_harmonic_freq, harmonic_freq, harmonic_SNR
                obj_func_val = {'Err_total': Err_total, 'Err_total_fine': Err_total_fine, 'freq_search': freq_search, \
                           'freq_search_fine': freq_search_fine}
                return fund_harmonic_freq, harmonic_freq, harmonic_SNR, obj_func_val



    return [], [], [], []


'''
This code is run after line 250 to get a plot of parametric search for the correct wt.
import matplotlib.pyplot as plt
wt_idx = 0
wt_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
wt_legends = [f"wt = {wt}" for wt in wt_range]
Err_total_wt = {}
plt.figure(wt_idx)
for wt in wt_range:
    Err_total_wt[wt_idx] = Error_ObjectiveFunctionEstimate(a_est, f_est, wt, freq_search, p1, p2, config_dict)
    
    plt.plot(freq_search, Err_total_wt[wt_idx])
    wt_idx = wt_idx + 1
plt.legend(wt_legends)
'''