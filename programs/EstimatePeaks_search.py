from scipy.signal import find_peaks, peak_prominences
import numpy as np
from scipy.signal.windows import kaiser
import yaml
from scipy.signal import argrelextrema
#/home/vesathya/Emanations/Journal/Emanations_JournalCode/Emanations/synapse_emanation.yaml
# def config_EstimatePeaks(config):
#     #"./Emanations/PythonScripts/synapse_emanation.yaml"

#     with open(yaml_filename, 'r') as config_file:
#         config = yaml.safe_load(config_file)
# #     global p1, p2, num_slice_ns, ns_estimate_percentile, Maxpeaks, gb_thresh, ntimes_ns, pct_samp_NF, NF_prctile
#     ########################################################################################################################
#     ########################################################################################################################
#     # Config file choices

#     #Noise spread calculation parameters:
#     # Per small slice, we take difference of p1 percentile and p2 percentile of SNRs of all points in that slice.
#     p1 = config['EstimatePeaks']['p1']
#     p2 = config['EstimatePeaks']['p2']
#     num_slice_ns = config['EstimatePeaks']['num_slice_ns'] # number of slices we create from an FFT. Large no. of slices helps deal with curved noise floor.
#     ns_estimate_percentile = config['EstimatePeaks']['ns_estimate_percentile'] # We take 50% of noise spread estimates from each of the slices, to get a single estimate.


#     # Peak estimation parameters:
#     # Maxpeaks = config['EstimatePeaks']['Maxpeaks'] # Top 200 peaks we identify from the gives FFT and process further upon them.
#     # gb_thresh = config['EstimatePeaks']['gb_thresh'] # Threshold we use to detect peaks that are a minimum of 1 + noise_spread.
#     # ntimes_ns = config['EstimatePeaks']['ntimes_ns'] # The way we use this ntimes_ns is as follows: thresh = np.min([ns + gb_thresh, ntimes_ns*ns])

#     # Localized noise floor for given peak parameters
#     pct_samp_NF = config['EstimatePeaks']['pct_samp_NF'] # we search around 10% of the samples around a peak
#     NF_prctile = config['EstimatePeaks']['NF_prctile'] # We look at 50% of the samples as the estimate
    
#     return p1, p2, num_slice_ns, ns_estimate_percentile, pct_samp_NF, NF_prctile
    ########################################################################################################################
    ########################################################################################################################

# (p1, p2, num_slice_ns, ns_estimate_percentile, pct_samp_NF, NF_prctile) = config_EstimatePeaks()


# Incoherent averaing over ABS of FFT done over 1 ms epochs to improve SNR.
def WelchPSDEstimate(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta,config_dict):
    # iq_feature = np.real(np.multiply(iq, np.conj(iq)))
    # shift = float(int(dur_ensemble*fs*(100-perc_overlap)/100))
    win_len = np.floor(dur_ensemble*fs).astype(int)
    shift = np.floor(win_len*((100-perc_overlap)/100)).astype(int)
    start_idx = 0
    if len(iq_feature) > win_len:
        end_idx = win_len
        inc_ens_avg_power = np.zeros(win_len)

        w = kaiser(int(win_len), kaiser_beta)
        w /= np.sum(w)
        w_energy = (np.real(np.vdot(w, w))) / len(w)
    else:
        end_idx = len(iq_feature)
        inc_ens_avg_power = np.zeros(end_idx)

        w = kaiser(int(end_idx), kaiser_beta)
        w /= np.sum(w)
        w_energy = (np.real(np.vdot(w, w))) / len(w)
    num_ensembles = 0

    while end_idx <= len(iq_feature):
        iqpower_ensemble = iq_feature[start_idx:end_idx]
        iqpower_win = np.multiply(iqpower_ensemble, w)
        inc_ensemble_fft = np.fft.fftshift(np.abs(np.fft.fft(iqpower_win)))
        inc_ensemble_power = np.multiply(inc_ensemble_fft, inc_ensemble_fft) / (w_energy * len(w))

        inc_ens_avg_power += inc_ensemble_power
        start_idx += shift
        end_idx += shift
        num_ensembles += 1
    ###################### Added by Hadi - commented by Venkatesh on May 11th
    #     if num_ensembles == 0:
    # #         w2 = kaiser(int(iq_feature), kaiser_beta)
    # #         w2 /= np.sum(w2)
    #         num_ensembles = 1
    #         iqpower_ensemble = iq_feature
    # #         iqpower_win = np.multiply(iqpower_ensemble, w2)
    #         inc_ens_avg_fft = np.fft.fftshift(np.abs(np.fft.fft(iqpower_ensemble)))
    ######################
    inc_ens_avg_power /= num_ensembles
    print("num_ensembles: ",num_ensembles)
    return inc_ens_avg_power

# Noise spread esimated piece wise and averaged to get a single estimate of noise spread
# for a non-flat noise floor, using a robust percentile methoid that is resilient to
# non-flat noise floor and outliers - contributed by spurts of high energy signal.
def Noisespread(fft_iq_dB,config_dict):
    # (p1 - p2) represent 68% -- one sigma around the mean.


    thresh_arr = np.zeros(config_dict['EstimatePeaks']['num_slice_ns'])
    start_idx = 0
    len_slice = np.floor(len(fft_iq_dB)/config_dict['EstimatePeaks']['num_slice_ns']).astype(int)
    end_idx = len_slice
    bins, nf = [], []
    # bin.append(start_idx)
    for i in range(0, config_dict['EstimatePeaks']['num_slice_ns']):
        thresh_arr[i] = np.percentile(fft_iq_dB[start_idx: end_idx], config_dict['EstimatePeaks']['p1']) - \
            np.percentile(fft_iq_dB[start_idx: end_idx], config_dict['EstimatePeaks']['p2'])
        NF_temp = np.percentile(fft_iq_dB[start_idx: end_idx], config_dict['EstimatePeaks']['NF_prctile'])
        nf.append(NF_temp)
        bins.append(start_idx)
        if i==0:
            nf.append(NF_temp)
        start_idx += len_slice
        end_idx += len_slice
    # ns = np.mean(thresh_arr)
    bins.append(start_idx)
    ns = np.percentile(thresh_arr, config_dict['EstimatePeaks']['ns_estimate_percentile'])
    return ns, bins, nf

# Estimate noise floor customized manner - by finding it around a local region surronding the peak.
# using similar percentile estimate that is robust to non-flat noise and bursty energies.
def NoiseFloorEveryPeak(fft_iq_dB, locpeak, pct_samp_NF, NF_prctile):
    # for every peak found, the corresponding NF from region around the peak is found.
    # Localized noise floor is important in cases of curved noise figure
    num_NFest = np.floor(len(fft_iq_dB)*pct_samp_NF/100).astype(int)
    NF = np.zeros(len(locpeak))
    for idx in range(0, len(locpeak)):
        if locpeak[idx] < np.floor(num_NFest/2).astype(int):
            start_idx = 0
            end_idx = start_idx + num_NFest
        elif locpeak[idx] + np.floor(num_NFest/2).astype(int) >= len(fft_iq_dB)-3:
            end_idx = len(fft_iq_dB)
            start_idx = end_idx - num_NFest
        else:
            start_idx = locpeak[idx] - np.floor(num_NFest/2).astype(int)
            end_idx = start_idx + num_NFest
        NF[idx] = np.percentile(fft_iq_dB[start_idx: end_idx], NF_prctile)
    return NF

# Top N peaks identified, peak freq. and SNR estimated, that cross a certain threshold in SNR metric.
def EstimatePeaks(fft_iq_dB, f_range, Fb, min_peaks_detect, gb_thresh, ntimes_ns, Maxpeaks,config_dict):
    # Input
    # fft_iq_dB: FFT of IQ samples in dB scale (10log10(fft(iq)))
    # f_range: Range of frequencies
    # Fb: Frequency bin size

    
    ns, bins, nf = Noisespread(fft_iq_dB, config_dict)
    print("noise spread: ", ns)
    # This type of threshold calculation - we add a guard band thrshold on top of noise_spread of signal. If ns << GB, then we can pick ntimes_ns*ns.
    # We tried reducing the threshold to noise-spread - it ended up picking a lot of noisy peaks and results were not good.
    thresh = np.min([ns + gb_thresh, ntimes_ns*ns])
    OldMethodfindpeaks = False
    if OldMethodfindpeaks: # Only prominence based thresholding.
        locpeak, properties = find_peaks(fft_iq_dB, prominence=thresh)
        # We might sometimes get too many peaks, say in thousands that satisfy the criterion specified.
        # We therefore have a check to ensure that doesn't happen. Else, it might overburden the fundamental harmonic estimation
        # TWM procedure.
        # We check if there are minimum number of peaks. The FFT of real signal has a mirror image of the peaks
        #Therefore the true min_peaks in positive frequency region is twice of it plus the tone at DC.
        if len(locpeak) > 2*min_peaks_detect+1:
            Npeaks_actual = np.min([Maxpeaks, len(locpeak)])

            # location of peaks in terms of indices are probably not sorted.
            # we get indices from sorting the prominences, which are in ascending order by default
            # we reverse them to get descending order indices
            sort_idx = np.argsort(properties["prominences"])
            temp_locpeak = locpeak[sort_idx]
            locpeak = temp_locpeak[::-1]
            locpeaks = locpeak[0:Npeaks_actual]

            NF = NoiseFloorEveryPeak(fft_iq_dB, locpeaks, config_dict['EstimatePeaks']['pct_samp_NF'], config_dict['EstimatePeaks']['NF_prctile'])

            # Parabolic interpolation
            fft_iq = 10**(fft_iq_dB/10)
            f_est_diff, a_est_non_log = FreqEst_parabolic(fft_iq, locpeaks, Fb)
            f_est = f_range[locpeaks] + f_est_diff
            # NF_actual = NF[0:Npeaks_actual]
            SNR = 10*np.log10(a_est_non_log) - NF
            return SNR, f_est
        else:
            return [], []

    else:
        # Compute indices of local maxima pointss
        local_max_idx = np.array(argrelextrema(fft_iq_dB, np.greater)).squeeze()
        # Compute noisefloor for each of the local maxima
        # NF_localmax = NoiseFloorEveryPeak(fft_iq_dB, local_max_idx, pct_samp_NF, NF_prctile)
        temp3 = np.digitize(local_max_idx, bins) - 1
        NF_localmax_new = np.array(nf)[temp3]
        # SNR computation for each of the local maxima
        # SNR_localmax = fft_iq_dB[local_max_idx] - NF_localmax
        SNR_localmax_new = fft_iq_dB[local_max_idx] - NF_localmax_new
        # Pick top SNR valued n_peaks peaks, location captured in locpeaks.
        temp_sortidx = np.argsort(SNR_localmax_new)
        lm_sortidx = local_max_idx[temp_sortidx]
        n_peaks = np.min([len(lm_sortidx), Maxpeaks])
        lm_ascend_sortidx = lm_sortidx[::-1]
        locpeaks_SNRThresh = lm_ascend_sortidx[0:n_peaks]
        # Add prominence based pruning to include parameterless distance based separation, especially peaks that are part of the same peak.
        prominence_dict = peak_prominences(fft_iq_dB, locpeaks_SNRThresh)
        # prominence_SNRThresh_locpeaks_temp = prominence_dict[0]
        locpeaks = locpeaks_SNRThresh[np.argwhere(prominence_dict[0] > ns)]
        locpeaks = locpeaks.squeeze()
        if len(locpeaks) > 2 * min_peaks_detect + 1:
            fft_iq = 10 ** (fft_iq_dB / 10)
            f_est_diff, a_est_non_log = FreqEst_parabolic(fft_iq, locpeaks, Fb)
            f_est = f_range[locpeaks] + f_est_diff
            # NF_actual = NF[0:Npeaks_actual]
            # NF = NoiseFloorEveryPeak(fft_iq_dB, locpeaks, pct_samp_NF, NF_prctile)
            temp3 = np.digitize(locpeaks, bins) - 1
            NF_peaks = np.array(nf)[temp3]
            SNR = 10 * np.log10(a_est_non_log) - NF_peaks
            idx4 = np.argwhere(SNR > thresh)
            SNR_thresh = SNR[idx4]
            f_est_thresh = f_est[idx4]
            return SNR_thresh.squeeze(), f_est_thresh.squeeze(), locpeaks
        else:
            return [], [], []
# Parabolic estimation to get more accurate estimates of frequencies and SNR of the peaks
def FreqEst_parabolic(fft_iq, locpeaks, Fb):
    a_est_nonlog = np.zeros((len(locpeaks)))
    f_est_diff = np.zeros((len(locpeaks)))
    for i in np.arange(0,len(locpeaks)):
        a1 = fft_iq[locpeaks[i]-1]
        a2 = fft_iq[locpeaks[i]]
        a3 = fft_iq[locpeaks[i]+1]
        b = (a3-a1)/2
        c = a2
        f_peak = ((a1-a3)*Fb)/(2*(a1+a3-2*a2))
        f_est_diff[i] = f_peak
        a_est_nonlog[i] = c - ((b**2)/4)
    return f_est_diff, a_est_nonlog