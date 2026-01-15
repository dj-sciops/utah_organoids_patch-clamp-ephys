"""
Feature extraction using ipfx (modern replacement for AllenSDK 0.14.2).

This module provides the same interface as current_clamp_features.py but uses
the modern ipfx library instead of the vendored AllenSDK code.

Usage:
    from .ipfx_features import extract_istep_features
    cell_features, summary_features = extract_istep_features(data, start=0.55, end=1.55)
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor
from ipfx import subthresh_features as subf
from ipfx import spike_train_features as strf


def extract_istep_features(data, start, end, subthresh_min_amp=-100, n_subthres_sweeps=4,
                           sag_target=-100, suprathreshold_target_delta_v=15,
                           suprathreshold_target_delta_i=15,
                           latency_target_delta_i=5,
                           filter=10., dv_cutoff=5., max_interval=0.02, min_height=10,
                           min_peak=-20., thresh_frac=0.05, baseline_interval=0.1,
                           baseline_detect_thresh=0.3, spike_detection_delay=0.001,
                           adapt_avg_n_sweeps=3, adapt_first_n_ratios=2,
                           sag_range_left=-120, sag_range_right=-95):
    """
    Compute cellular ephys features from square pulse current injections.

    Uses ipfx library (modern replacement for AllenSDK 0.14.2).

    Parameters match the legacy function for compatibility.
    """
    if filter * 1000 >= data['hz']:
        filter = None

    t = data['t']
    n_sweeps = data['n_sweeps']
    voltage = data['voltage']
    current = data['current']

    # Extract spike features for each sweep
    spk_ext = SpikeFeatureExtractor(
        start=start, end=end,
        filter=filter, dv_cutoff=dv_cutoff,
        max_interval=max_interval, min_height=min_height,
        min_peak=min_peak, thresh_frac=thresh_frac
    )

    sweep_data = []
    all_spikes = []
    spikes_sweep_ids = []

    for sweep_idx in range(n_sweeps):
        v = voltage[sweep_idx]
        i = current[sweep_idx]

        # Calculate stimulus amplitude (max current during stimulus window)
        stim_mask = (t >= start) & (t <= end)
        stim_amp = np.mean(i[stim_mask])

        # Baseline voltage (before stimulus)
        baseline_mask = (t >= start - baseline_interval) & (t < start)
        v_baseline = np.mean(v[baseline_mask]) if np.any(baseline_mask) else np.mean(v[:int(start / data['dt'])])

        # Extract spikes for this sweep
        spk_df = spk_ext.process(t, v, i)
        n_spikes = len(spk_df)

        # Calculate average firing rate
        if n_spikes > 0:
            duration = end - start
            avg_rate = n_spikes / duration
        else:
            avg_rate = 0.0

        sweep_info = {
            'id': sweep_idx,
            'stim_amp': stim_amp,
            'v_baseline': v_baseline,
            'n_spikes': n_spikes,
            'avg_rate': avg_rate,
            'spikes': spk_df.to_dict('records') if n_spikes > 0 else [],
            'spikes_df': spk_df,
        }

        # Additional spike train features if spikes present
        if n_spikes > 0:
            # Latency to first spike
            sweep_info['latency'] = float(spk_df.iloc[0]['threshold_t'] - start)

            # ISI statistics
            if n_spikes > 1:
                isis = np.diff(spk_df['peak_t'].values)
                sweep_info['median_isi'] = float(np.median(isis))
                sweep_info['first_isi'] = float(isis[0])

                # Adaptation index
                if n_spikes >= 3:
                    sweep_info['adapt'] = _calculate_adaptation_index(isis)
                else:
                    sweep_info['adapt'] = np.nan
            else:
                sweep_info['median_isi'] = np.nan
                sweep_info['first_isi'] = np.nan
                sweep_info['adapt'] = np.nan

            # Collect all spikes with sweep ID
            for spike in sweep_info['spikes']:
                all_spikes.append(spike)
                spikes_sweep_ids.append(sweep_idx)
        else:
            sweep_info['latency'] = np.nan
            sweep_info['median_isi'] = np.nan
            sweep_info['first_isi'] = np.nan
            sweep_info['adapt'] = np.nan

        sweep_data.append(sweep_info)

    # Identify spiking and subthreshold sweeps
    spiking_sweeps = [s for s in sweep_data if s['n_spikes'] > 0]
    subthreshold_sweeps = [s for s in sweep_data if s['n_spikes'] == 0]

    # Sort by stimulus amplitude
    spiking_sweeps = sorted(spiking_sweeps, key=lambda x: x['stim_amp'])
    subthreshold_sweeps = sorted(subthreshold_sweeps, key=lambda x: x['stim_amp'])

    # Find rheobase (first spiking sweep)
    if spiking_sweeps:
        has_AP = True
        rheobase_sweep = spiking_sweeps[0]
        rheobase_i = rheobase_sweep['stim_amp']
        rheobase_index = rheobase_sweep['id']
        first_spike = rheobase_sweep['spikes'][0] if rheobase_sweep['spikes'] else {}
    else:
        has_AP = False
        rheobase_sweep = None
        rheobase_i = None
        rheobase_index = None
        first_spike = {}

    # Calculate input resistance from subthreshold sweeps
    input_resistance, input_resistance_vm, input_resistance_stim_ap = _calculate_input_resistance(
        data, subthreshold_sweeps, start, end, baseline_interval,
        subthresh_min_amp, n_subthres_sweeps
    )

    # Calculate membrane time constant and capacitance
    tau = _calculate_tau(data, subthreshold_sweeps, start, end, baseline_interval)
    if tau is not None and input_resistance is not None and input_resistance > 0:
        capacitance = tau / input_resistance * 1e6  # pF
    else:
        capacitance = None

    # Calculate sag
    sag, vm_for_sag, sag_sweeps, indices_for_sag = _calculate_sag(
        data, subthreshold_sweeps, start, end,
        sag_target, sag_range_left, sag_range_right
    )

    # Calculate F-I curve slope (only using spiking sweeps with positive rates)
    if len(spiking_sweeps) >= 1:
        # Sort by stimulus amplitude to ensure correct ordering
        sorted_spiking = sorted(spiking_sweeps, key=lambda x: x['stim_amp'])
        stim_amps = [s['stim_amp'] for s in sorted_spiking]
        rates = [s['avg_rate'] for s in sorted_spiking]

        # Find the last subthreshold sweep amplitude
        last_subthres_amp = None
        if subthreshold_sweeps:
            # Get the subthreshold sweep with highest (least negative) current below rheobase
            sorted_subthres = sorted(subthreshold_sweeps, key=lambda x: x['stim_amp'])
            if sorted_subthres:
                last_subthres_amp = sorted_subthres[-1]['stim_amp']

        fi_fit_slope = _fit_fi_slope(stim_amps, rates, last_subthres_amp=last_subthres_amp)
    else:
        fi_fit_slope = None

    # Calculate baseline voltage
    v_baseline = np.mean([s['v_baseline'] for s in subthreshold_sweeps]) if subthreshold_sweeps else \
                 (sweep_data[0]['v_baseline'] if sweep_data else None)

    # Bias current (assumed 0 for these protocols)
    bias_current = 0.0

    # Hero sweep selection
    hero_sweep = None
    hero_sweep_stim_amp = None
    hero_sweep_index = None
    avg_rate = None
    avg_hs_latency = None
    avg_rheobase_latency = None

    if has_AP:
        hero_stim_target = rheobase_i + suprathreshold_target_delta_i - 1
        latency_stim_target = rheobase_i + latency_target_delta_i

        # Find hero sweep
        last_sweep = None
        for sweep in spiking_sweeps:
            if sweep['stim_amp'] > hero_stim_target:
                hero_sweep = sweep
                break
            last_sweep = sweep

        if hero_sweep and last_sweep:
            hero_amp = hero_sweep['stim_amp']
            pre_hero_amp = last_sweep['stim_amp']
            hs_latency = hero_sweep.get('latency', np.nan)
            pre_hs_latency = last_sweep.get('latency', np.nan)
            hs_rate = hero_sweep['avg_rate']
            pre_hs_rate = last_sweep['avg_rate']

            if hero_amp != pre_hero_amp:
                avg_hs_latency = ((hero_amp - hero_stim_target) * pre_hs_latency +
                                  (hero_stim_target - pre_hero_amp) * hs_latency) / (hero_amp - pre_hero_amp)
                avg_rate = ((hero_amp - hero_stim_target) * pre_hs_rate +
                            (hero_stim_target - pre_hero_amp) * hs_rate) / (hero_amp - pre_hero_amp)

            hero_sweep_stim_amp = hero_sweep['stim_amp']
            hero_sweep_index = hero_sweep['id']
        elif last_sweep:
            avg_hs_latency = last_sweep.get('latency', np.nan)
            avg_rate = last_sweep['avg_rate']
            print("Could not find hero sweep.")

        # Find latency sweep
        last_latency_sweep = None
        latency_sweep = None
        for sweep in spiking_sweeps:
            if sweep['stim_amp'] > latency_stim_target:
                latency_sweep = sweep
                break
            last_latency_sweep = sweep

        if latency_sweep and last_latency_sweep:
            latency_amp = latency_sweep['stim_amp']
            pre_latency_amp = last_latency_sweep['stim_amp']
            latency_above = latency_sweep.get('latency', np.nan)
            latency_below = last_latency_sweep.get('latency', np.nan)

            if latency_amp != pre_latency_amp:
                avg_rheobase_latency = ((latency_amp - latency_stim_target) * latency_below +
                                        (latency_stim_target - pre_latency_amp) * latency_above) / (latency_amp - pre_latency_amp)
        elif last_latency_sweep:
            avg_rheobase_latency = last_latency_sweep.get('latency', np.nan)

    # Max firing rate
    max_firing_rate = max([s['avg_rate'] for s in sweep_data]) if sweep_data else 0.0

    # Custom adaptation calculation
    spikes_peak_t = np.array([s['peak_t'] for s in all_spikes]) if all_spikes else np.array([])
    spikes_sweep_id = np.array(spikes_sweep_ids)
    adapt_avg, adapt_all = _calculate_adapt(
        spikes_sweep_id, spikes_peak_t, start,
        adapt_interval=1.0, max_isi_ratio=2.5, min_peaks=4,
        avg_n_sweeps=adapt_avg_n_sweeps, first_n_adapt_ratios=adapt_first_n_ratios
    )

    # Build cell_features dict (mimics legacy structure)
    cell_features = {
        'v_baseline': v_baseline,
        'bias_current': bias_current,
        'tau': tau,
        'input_resistance': input_resistance,
        'input_resistance_vm': input_resistance_vm,
        'input_resistance_stim_ap': input_resistance_stim_ap,
        'fi_fit_slope': fi_fit_slope,
        'sag': sag,
        'vm_for_sag': vm_for_sag,
        'sag_sweeps': sag_sweeps,
        'indices_for_sag': indices_for_sag,
        'rheobase_i': rheobase_i,
        'rheobase_extractor_index': rheobase_index,
        'rheobase_sweep': rheobase_sweep,
        'sweeps': sweep_data,
        'spiking_sweeps': spiking_sweeps,
        'hero_sweep': hero_sweep if hero_sweep else {},
        'hero_sweep_stim_amp': hero_sweep_stim_amp,
        'hero_sweep_index': hero_sweep_index,
        'hero_sweep_stim_target': hero_stim_target if has_AP else None,
        'first_spike': first_spike,
    }

    # Build summary_features (same structure as legacy)
    summary_features = OrderedDict([
        ('file_id', data['file_id']),
        ('has_ap', has_AP),
        ('v_baseline', v_baseline),
        ('bias_current', bias_current),
        ('tau', tau * 1000 if tau is not None else None),  # Convert to ms
        ('capacitance', capacitance),
        ('membrane_cap', capacitance),  # Alias for backward compatibility
        ('input_resistance', input_resistance),
        ('f_i_curve_slope', fi_fit_slope),
        ('max_firing_rate', max_firing_rate),
        ('sag', sag),
        ('vm_for_sag', vm_for_sag),
        ('indices_for_sag', indices_for_sag),
        ('sag_sweep_indices', sag_sweeps),
        ('ap_threshold', first_spike.get('threshold_v')),
        ('ap_width', first_spike.get('width') * 1000 if first_spike.get('width') is not None else None),
        ('ap_height', first_spike['peak_v'] - first_spike['trough_v'] if has_AP and 'peak_v' in first_spike else None),
        ('ap_peak', first_spike.get('peak_v')),
        ('ap_trough', first_spike.get('trough_v')),
        ('ap_fast_trough', first_spike.get('fast_trough_v')),
        ('ap_slow_trough', first_spike.get('slow_trough_v')),
        ('ap_adp', first_spike.get('adp_v')),
        ('ap_trough_3w', first_spike.get('trough_3w_v')),
        ('ap_trough_4w', first_spike.get('trough_4w_v')),
        ('ap_trough_5w', first_spike.get('trough_5w_v')),
        ('ap_trough_to_threshold', first_spike['threshold_v'] - first_spike['trough_v'] if has_AP and 'threshold_v' in first_spike else None),
        ('ap_trough_4w_to_threshold', first_spike['threshold_v'] - first_spike.get('trough_4w_v') if has_AP and 'threshold_v' in first_spike and first_spike.get('trough_4w_v') else None),
        ('ap_trough_5w_to_threshold', first_spike['threshold_v'] - first_spike.get('trough_5w_v') if has_AP and 'threshold_v' in first_spike and first_spike.get('trough_5w_v') else None),
        ('ap_peak_to_threshold', first_spike['peak_v'] - first_spike['threshold_v'] if has_AP and 'peak_v' in first_spike else None),
        ('ap_upstroke', first_spike.get('upstroke')),
        ('ap_downstroke', -first_spike.get('downstroke') if has_AP and first_spike.get('downstroke') else None),
        ('ap_updownstroke_ratio', first_spike.get('upstroke_downstroke_ratio')),
        ('hs_firing_rate', hero_sweep['avg_rate'] if hero_sweep else None),
        ('avg_firing_rate', avg_rate),
        ('hs_adaptation', hero_sweep.get('adapt') if hero_sweep else None),
        ('hs_median_isi', hero_sweep.get('median_isi') if hero_sweep else None),
        ('hs_latency', hero_sweep.get('latency') * 1000 if hero_sweep and hero_sweep.get('latency') else None),
        ('avg_hs_latency', avg_hs_latency * 1000 if avg_hs_latency is not None else None),
        ('avg_rheobase_latency', avg_rheobase_latency * 1000 if avg_rheobase_latency is not None else None),
        ('first_spike_latency', avg_rheobase_latency * 1000 if avg_rheobase_latency is not None else None),
        ('rheobase_index', rheobase_index),
        ('rheobase_stim_amp', rheobase_i),
        ('hero_sweep_stim_amp', hero_sweep_stim_amp),
        ('hero_sweep_index', hero_sweep_index),
        ('all_firing_rate', np.array([s['avg_rate'] for s in sweep_data])),
        ('all_stim_amp', np.array([s['stim_amp'] for s in sweep_data])),
        ('input_resistance_vm', input_resistance_vm),
        ('input_resistance_stim_ap', input_resistance_stim_ap),
        ('all_adaptation', np.array([s.get('adapt', np.nan) for s in sweep_data])),
        ('all_v_baseline', np.array([s['v_baseline'] for s in sweep_data])),
        ('all_median_isi', np.array([s.get('median_isi', np.nan) for s in sweep_data])),
        ('all_first_isi', np.array([s.get('first_isi', np.nan) for s in sweep_data])),
        ('all_latency', np.array([s.get('latency', np.nan) for s in sweep_data])),
        ('spikes_sweep_id', spikes_sweep_id),
        ('spikes_threshold_t', np.array([s['threshold_t'] for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_peak_t', spikes_peak_t),
        ('spikes_trough_t', np.array([s['trough_t'] for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_threshold_v', np.array([s['threshold_v'] for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_peak_v', np.array([s['peak_v'] for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_trough_v', np.array([s['trough_v'] for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_fast_trough_t', np.array([s.get('fast_trough_t') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_fast_trough_v', np.array([s.get('fast_trough_v') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_slow_trough_t', np.array([s.get('slow_trough_t') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_slow_trough_v', np.array([s.get('slow_trough_v') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_adp_t', np.array([s.get('adp_t') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_adp_v', np.array([s.get('adp_v') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_trough_3w_t', np.array([s.get('trough_3w_t') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_trough_3w_v', np.array([s.get('trough_3w_v') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_trough_4w_t', np.array([s.get('trough_4w_t') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_trough_4w_v', np.array([s.get('trough_4w_v') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_trough_5w_t', np.array([s.get('trough_5w_t') for s in all_spikes]) if all_spikes else np.array([])),
        ('spikes_trough_5w_v', np.array([s.get('trough_5w_v') for s in all_spikes]) if all_spikes else np.array([])),
        ('adapt_avg', adapt_avg),
        ('adaptation_index', adapt_avg),  # Alias
        ('resting_vm', v_baseline),  # Alias
    ])

    return cell_features, summary_features


def _calculate_adaptation_index(isis):
    """Calculate adaptation index from ISIs."""
    if len(isis) < 2:
        return np.nan

    ratios = []
    for i in range(1, len(isis)):
        a, b = isis[i], isis[i-1]
        if a + b > 0:
            ratios.append((a - b) / (a + b))

    return np.mean(ratios) if ratios else np.nan


def _calculate_input_resistance(data, subthreshold_sweeps, start, end, baseline_interval,
                                 subthresh_min_amp, n_subthres_sweeps):
    """Calculate input resistance from subthreshold sweeps."""
    if not subthreshold_sweeps:
        return None, None, None

    # Filter sweeps by minimum amplitude
    valid_sweeps = [s for s in subthreshold_sweeps if s['stim_amp'] >= subthresh_min_amp]

    if len(valid_sweeps) < 2:
        return None, None, None

    # Use first n sweeps
    sweeps_to_use = valid_sweeps[:n_subthres_sweeps]

    t = data['t']
    stim_amps = []
    voltage_deflections = []

    for sweep in sweeps_to_use:
        idx = sweep['id']
        v = data['voltage'][idx]

        # Baseline voltage
        baseline_mask = (t >= start - baseline_interval) & (t < start)
        v_baseline = np.mean(v[baseline_mask])

        # Steady-state voltage during stimulus (last 100ms)
        ss_start = end - 0.1
        ss_mask = (t >= ss_start) & (t <= end)
        v_ss = np.mean(v[ss_mask])

        stim_amps.append(sweep['stim_amp'])
        voltage_deflections.append(v_ss - v_baseline)

    # Linear fit: dV = Rin * dI
    # Rin in MOhm = dV (mV) / dI (pA) * 1000
    stim_amps = np.array(stim_amps)
    voltage_deflections = np.array(voltage_deflections)

    # Only use negative current steps for input resistance
    neg_mask = stim_amps < 0
    if np.sum(neg_mask) < 2:
        neg_mask = np.ones(len(stim_amps), dtype=bool)

    if np.sum(neg_mask) >= 2:
        slope, _ = np.polyfit(stim_amps[neg_mask], voltage_deflections[neg_mask], 1)
        input_resistance = slope * 1000  # MOhm
    else:
        input_resistance = None

    # Return vm and stim_amp arrays for reference
    input_resistance_vm = voltage_deflections.tolist()
    input_resistance_stim_ap = stim_amps.tolist()

    return input_resistance, input_resistance_vm, input_resistance_stim_ap


def _calculate_tau(data, subthreshold_sweeps, start, end, baseline_interval):
    """Calculate membrane time constant from subthreshold sweeps."""
    if not subthreshold_sweeps:
        return None

    # Use the first negative-going sweep
    neg_sweeps = [s for s in subthreshold_sweeps if s['stim_amp'] < 0]
    if not neg_sweeps:
        return None

    sweep = neg_sweeps[0]
    idx = sweep['id']
    t = data['t']
    v = data['voltage'][idx]

    # Baseline voltage
    baseline_mask = (t >= start - baseline_interval) & (t < start)
    v_baseline = np.mean(v[baseline_mask])

    # Find the initial response region (first 200ms after stimulus start)
    tau_window = 0.2
    tau_mask = (t >= start) & (t <= start + tau_window)

    t_fit = t[tau_mask] - start
    v_fit = v[tau_mask] - v_baseline

    try:
        from scipy.optimize import curve_fit

        def exp_decay(x, a, tau):
            return a * (1 - np.exp(-x / tau))

        # Initial guess
        v_ss = v_fit[-1]
        popt, _ = curve_fit(exp_decay, t_fit, v_fit, p0=[v_ss, 0.02], maxfev=5000)
        tau = popt[1]

        if tau <= 0 or tau > 1:  # Sanity check
            return None

        return tau
    except Exception:
        return None


def _calculate_sag(data, subthreshold_sweeps, start, end, sag_target, sag_range_left, sag_range_right):
    """Calculate sag from hyperpolarizing sweeps."""
    if not subthreshold_sweeps:
        return None, None, None, None

    t = data['t']

    # Get voltage minima for all subthreshold sweeps
    sweep_mins = []
    for sweep in subthreshold_sweeps:
        idx = sweep['id']
        v = data['voltage'][idx]

        # Find minimum voltage during stimulus
        stim_mask = (t >= start) & (t <= end)
        v_min = np.min(v[stim_mask])
        sweep_mins.append((sweep, v_min))

    # First try to find sweeps in the sag range
    sag_sweeps_list = [(s, v) for s, v in sweep_mins if sag_range_left <= v <= sag_range_right]

    # If no sweeps in range, use the most negative hyperpolarizing sweep
    if not sag_sweeps_list:
        # Filter to only negative-going sweeps
        negative_sweeps = [(s, v) for s, v in sweep_mins if s['stim_amp'] < 0]
        if not negative_sweeps:
            return None, None, None, None
        # Use the sweep with most negative voltage (closest to sag_target behavior)
        sag_sweeps_list = [min(negative_sweeps, key=lambda x: x[1])]

    # Use the sweep closest to target
    sag_sweeps_list.sort(key=lambda x: abs(x[1] - sag_target))
    best_sweep, vm_for_sag = sag_sweeps_list[0]
    idx = best_sweep['id']
    v = data['voltage'][idx]

    # Calculate sag
    stim_mask = (t >= start) & (t <= end)
    v_stim = v[stim_mask]
    t_stim = t[stim_mask]

    # Peak (minimum) voltage
    v_peak = np.min(v_stim)
    peak_idx = np.argmin(v_stim)

    # Steady-state voltage (last 100ms)
    ss_start = end - 0.1
    ss_mask = (t_stim >= ss_start)
    v_ss = np.mean(v_stim[ss_mask])

    # Baseline voltage
    baseline_mask = (t >= start - 0.1) & (t < start)
    v_baseline = np.mean(v[baseline_mask])

    # Sag = (v_ss - v_peak) / (v_baseline - v_peak)
    denom = v_baseline - v_peak
    if abs(denom) > 0.1:  # Avoid division by zero
        sag = (v_ss - v_peak) / denom
    else:
        sag = 0.0

    sag_sweeps = [idx]
    indices_for_sag = [peak_idx]

    return sag, vm_for_sag, sag_sweeps, indices_for_sag


def _fit_fi_slope(stim_amps, rates, last_subthres_amp=None):
    """
    Fit F-I curve slope using the legacy algorithm.

    Key features:
    1. Include the last subthreshold sweep (rate=0) in the fit
    2. Only fit data up to the maximum firing rate
    """
    if len(stim_amps) < 1:
        return None

    stim_amps = np.array(stim_amps)
    rates = np.array(rates)

    # Insert the last subthreshold sweep (rate=0) at the beginning
    if last_subthres_amp is not None:
        stim_amps = np.concatenate([[last_subthres_amp], stim_amps])
        rates = np.concatenate([[0.0], rates])

    # Only fit up to the maximum firing rate
    # (sometimes high current injection stops the cell from firing)
    max_idx = np.argmax(rates)
    max_rate = rates[max_idx]

    if max_rate < 2:
        return 0.0

    stim_amps = stim_amps[:max_idx + 1]
    rates = rates[:max_idx + 1]

    if len(stim_amps) < 2:
        return None

    # Linear fit
    A = np.vstack([stim_amps, np.ones_like(stim_amps)]).T
    m, c = np.linalg.lstsq(A, rates, rcond=None)[0]

    return m


def _calculate_adapt(spikes_sweep_id, spikes_peak_t, start, end=None, adapt_interval=1.0,
                     min_peaks=4, max_isi_ratio=2.5, avg_n_sweeps=3, first_n_adapt_ratios=None,
                     firing_rate_target=None):
    """
    Calculate adaptation ratio across sweeps.

    This matches the legacy implementation for compatibility.
    """
    if len(spikes_sweep_id) == 0:
        return None, None

    end_adapt = start + adapt_interval
    mask = spikes_peak_t < end_adapt
    sweep_id = spikes_sweep_id[mask]
    peaks_all = spikes_peak_t[mask]

    # Group peaks by sweep
    peaks = {}
    for k, v in zip(sweep_id, peaks_all):
        if k in peaks:
            peaks[k].append(v)
        else:
            peaks[k] = [v]

    # Filter sweeps with insufficient spikes
    peaks = {k: v for k, v in peaks.items() if len(v) >= min_peaks}
    if not peaks:
        return None, None

    # Calculate ISIs
    isi = {k: [x - y for x, y in zip(v[1:], v[:-1])] for k, v in peaks.items()}

    # Filter out long intervals
    for k, v in isi.items():
        for i in range(1, len(v)):
            if v[i] > v[i-1] * max_isi_ratio:
                isi[k] = v[:i]
                break
            elif v[i-1] > v[i] * max_isi_ratio:
                isi[k] = v[:i-1]
                break

    # Filter sweeps with insufficient ISIs
    isi = {k: v for k, v in isi.items() if len(v) >= min_peaks - 1}
    if not isi:
        return None, None

    # Use only first n sweeps
    if len(isi) > avg_n_sweeps:
        keys = sorted(list(isi.keys()))[:avg_n_sweeps]
        isi = {k: isi[k] for k in keys}

    # Calculate adaptation ratios
    adapt = {}
    for k, v in isi.items():
        adapt[k] = [(x - y) / (x + y) for x, y in zip(v[1:], v[:-1])]

    # Average adaptation
    adapt_all = [np.mean(adapt[k][:first_n_adapt_ratios]) for k in adapt]
    adapt_mean = np.mean(adapt_all)

    return adapt_mean, adapt_all
