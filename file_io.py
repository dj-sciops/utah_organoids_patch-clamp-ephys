
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
import gzip

try:
    import cPickle as pickle  # much faster
except:
    import pickle

# Modern ABF reading library (preferred)
try:
    import pyabf
    HAS_PYABF = True
except ImportError:
    HAS_PYABF = False

# Legacy ABF reading library (fallback)
try:
    import stfio
    from stfio import StfIOException
    HAS_STFIO = True
except ImportError:
    HAS_STFIO = False
    if not HAS_PYABF:
        print("Neither pyabf nor stfio is installed. Make sure .abf files are converted to .pkl.")


def _load_current_step_pyabf(abf_file, min_voltage=-140):
    """
    Load current clamp recordings using pyabf (modern library).

    Key insight: For current clamp, we need the COMMAND current (stimulus injected
    into the cell), not the measured current. The command waveform is on the DAC
    channel, accessed via abf.sweepC after setting the appropriate channel.

    Returns data dict with same structure as legacy stfio loader.
    """
    abf = pyabf.ABF(abf_file)

    # Find voltage channel (ADC with mV units)
    voltage_ch = None
    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        if abf.sweepUnitsY == 'mV' or 'mV' in abf.sweepUnitsY:
            voltage_ch = ch
            break

    if voltage_ch is None:
        raise ValueError(f"No voltage channel (mV) found in {abf_file}")

    # Find current command channel (DAC with pA units)
    # The command waveform is what we injected, not what we measured
    current_cmd_ch = None
    for dac_idx, unit in enumerate(abf.dacUnits):
        if unit == 'pA' or 'pA' in unit:
            current_cmd_ch = dac_idx
            break

    if current_cmd_ch is None:
        raise ValueError(f"No current command channel (pA) found in DAC channels of {abf_file}")

    # Build data structure
    data = OrderedDict()
    data['file_id'] = os.path.splitext(os.path.basename(abf_file))[0]
    data['file_directory'] = os.path.dirname(abf_file)
    data['record_date'] = abf.abfDateTime.date()
    data['record_time'] = abf.abfDateTime.time()

    data['dt'] = 1.0 / abf.sampleRate  # seconds per sample
    data['hz'] = abf.sampleRate
    data['time_unit'] = 's'

    data['n_channels'] = abf.channelCount
    data['channel_names'] = abf.adcNames[:2] if len(abf.adcNames) >= 2 else abf.adcNames
    data['channel_units'] = abf.adcUnits[:2] if len(abf.adcUnits) >= 2 else abf.adcUnits
    data['n_sweeps'] = abf.sweepCount
    data['sweep_length'] = abf.sweepPointCount

    data['t'] = np.arange(0, data['sweep_length']) * data['dt']

    # Load voltage and current command data for each sweep
    voltage_list = []
    current_list = []

    for sweep_idx in range(abf.sweepCount):
        # Get voltage trace
        abf.setSweep(sweep_idx, channel=voltage_ch)
        voltage_data = abf.sweepY.copy()

        # Get current command (stimulus) from DAC
        # Must set sweep with the correct channel to access sweepC
        abf.setSweep(sweep_idx, channel=current_cmd_ch)
        current_data = abf.sweepC.copy()

        voltage_list.append(voltage_data)
        current_list.append(current_data)

    data['voltage'] = voltage_list
    data['current'] = current_list

    # Filter out sweeps with voltage below threshold
    if min_voltage is not None:
        to_pop = []
        for i, v in enumerate(data['voltage']):
            if np.min(v) < min_voltage:
                to_pop.append(i)
        if to_pop:
            data['voltage'] = [v for i, v in enumerate(data['voltage']) if i not in to_pop]
            data['current'] = [c for i, c in enumerate(data['current']) if i not in to_pop]
            data['n_sweeps'] -= len(to_pop)

    return data


def _load_current_step_stfio(abf_file, channels=[0, 1], min_voltage=-140):
    """
    Load current clamp recordings using stfio (legacy library).

    Note: This reads the MEASURED current from ADC, not the command current.
    For most analyses, the pyabf loader is preferred as it reads the actual
    stimulus command from the DAC channel.
    """
    rec = stfio.read(str(abf_file))  # Explicit str() for Path compatibility
    ch0, ch1 = channels[0], channels[1]

    assert((rec[ch0].yunits in ['mV', 'pA']) and (rec[ch1].yunits in ['mV', 'pA']))

    data = OrderedDict()
    data['file_id'] = os.path.splitext(os.path.basename(abf_file))[0]
    data['file_directory'] = os.path.dirname(abf_file)
    data['record_date'] = rec.datetime.date()
    data['record_time'] = rec.datetime.time()

    data['dt'] = rec.dt / 1000
    data['hz'] = 1./rec.dt * 1000
    data['time_unit'] = 's'

    data['n_channels'] = len(rec)
    data['channel_names'] = [rec[x].name for x in channels]
    data['channel_units'] = [rec[x].yunits for x in channels]
    data['n_sweeps'] = len(rec[ch0])
    data['sweep_length'] = len(rec[ch0][0])

    data['t'] = np.arange(0, data['sweep_length']) * data['dt']

    if rec[ch0].yunits == 'mV' and rec[ch1].yunits == 'pA':
        data['voltage'] = rec[ch0]
        data['current'] = rec[ch1]
    elif rec[ch1].yunits == 'mV' and rec[ch0].yunits == 'pA':
        data['voltage'] = rec[ch1]
        data['current'] = rec[ch0]
    else:
        raise ValueError("channel y-units must be 'mV' or 'pA'.")
    data['voltage'] = [x.asarray() for x in data['voltage']]
    data['current'] = [x.asarray() for x in data['current']]

    if min_voltage is not None:
        to_pop = []
        for i, x in enumerate(data['voltage']):
            if np.min(x) < min_voltage:
                to_pop.append(i)
        data['voltage'] = [x for i, x in enumerate(data['voltage']) if i not in to_pop]
        data['current'] = [x for i, x in enumerate(data['current']) if i not in to_pop]
        data['n_sweeps'] -= len(to_pop)

    return data


def load_current_step(abf_file, filetype='abf', channels=[0, 1], min_voltage=-140):
    """
    Load current clamp recordings from pClamp .abf files.

    Parameters
    ----------
    abf_file : str
        Path to ABF file
    filetype : str
        One of ['abf', 'pkl']. 'pkl' is pickle file converted from abf.
    channels : list
        Channel indices [voltage_ch, current_ch] for stfio fallback.
        Not used by pyabf loader which auto-detects channels.
    min_voltage : float or None
        Traces with min below this voltage are excluded (e.g., -140).

    Returns
    -------
    data : OrderedDict
        Contains voltage, current, timing info, and metadata.

    Notes
    -----
    Uses pyabf (preferred) which reads command current from DAC.
    Falls back to stfio or pkl if pyabf unavailable or fails.
    """
    # Try pickle file first if requested
    if filetype == 'pkl':
        return _load_from_pickle(abf_file, filetype)

    # Try pyabf (preferred - reads command current correctly)
    if HAS_PYABF:
        try:
            return _load_current_step_pyabf(abf_file, min_voltage=min_voltage)
        except Exception as e:
            # Fall through to stfio or pkl
            pass

    # Try stfio (legacy fallback)
    if HAS_STFIO:
        try:
            return _load_current_step_stfio(abf_file, channels=channels, min_voltage=min_voltage)
        except Exception:
            pass

    # Final fallback: try pkl file
    return _load_from_pickle(abf_file, filetype)


def _load_from_pickle(abf_file, filetype):
    """Load data from pickle file (legacy format)."""
    pkl_file = os.path.splitext(abf_file)[0] + '.pkl'

    try:
        with gzip.open(pkl_file, 'rb') as handle:
            data = pickle.load(handle, encoding='bytes')
    except (IOError, OSError):
        with open(pkl_file, 'rb') as handle:
            data = pickle.load(handle, encoding='bytes')

    return decode_bytes(data)


def decode_bytes(data):
    if isinstance(data, bytes):
        return data.decode('ascii')
    elif isinstance(data, dict):
        return dict(map(decode_bytes, data.items()))
    elif isinstance(data, tuple):
        return tuple(map(decode_bytes, data))
    elif isinstance(data, list):
        return list(map(decode_bytes, data))
    else:
        return data


def save_data_as_pickle(data, pkl_file, compress=True):
    if compress:
        with gzip.open(pkl_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=2)
    else:
        with open(pkl_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=2)

def current_clamp_abf_to_pkl(in_file, out_file):
    print("Saving: " + out_file)
    data = load_current_step(in_file, channels=[0,1], min_voltage=-140)
    save_data_as_pickle(data, out_file)

def batch_current_clamp_abf_to_pkl(input_folder, output_folder=None):
    if not output_folder is None and input_folder != output_folder:
        try:
            os.makedirs(output_folder)
        except OSError:
            pass
    if output_folder is None:
        output_folder = input_folder

    filenames = [x for x in os.listdir(input_folder) if x.endswith(".abf")]
    input_file_paths = [os.path.join(input_folder, x) for x in filenames]
    output_file_paths = [os.path.join(output_folder, x[:-4] + ".pkl") for x in filenames]
    for in_file, out_file in zip(input_file_paths, output_file_paths):
        current_clamp_abf_to_pkl(in_file, out_file)
