# ViRa24 logfile converter
# Copyright (C) 2026 M.A. @ NVISION
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.


##% packages

# 3rd party math packages
import pandas as pd
import numpy as np
from scipy.constants import c
from scipy.signal import butter, filtfilt, iirnotch

# sys, path and argparse libraries
import sys
from pathlib import Path
import argparse

# Add python project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# additional functions
from resources.vira_lib import estimate_offset

# Support function
def read_header(rf_file, verbose = True):
    ver = ""
    fmt = ""
    start_time_str = ""
    srate = None
    header_lines = []
    with open(rf_file, encoding="utf-8") as f:
        for line in f:
            newline = line.strip()
            if newline.startswith("#"):
                header_lines.append(newline.strip())
                if "UI version" in newline:
                    ver = float(newline.split("UI version ")[1])
                if "Format:" in newline:
                    fmt = newline.split("Format:")[1]
                if "Start time:" in newline: 
                    start_time_str = newline.split("Start time:")[1]
                if "Sample rate:" in newline: 
                    srate = float(newline.split("Sample rate:")[1].replace("Sa/s",""))

            else:
                if len(header_lines)<4:
                    raise Exception(f'No correct # header found in {rf_file}') 
    if verbose:
        print(f'Converted: {rf_file} file has been read')
        print(f'ver = {ver}, start_time_str = {start_time_str}, srate = {srate}')
        for line in header_lines:
            print(line)
    return header_lines, ver, fmt
        
        
# Classes defined at top
class StateRF:
    def __init__(self, buffer_len = 5000, ac_coupling = False):
        # logging and coupling flags
        self.ac_coupling = ac_coupling
        self.logging = True

        # Offset initialization
        self.offset_i = 0
        self.offset_q = 0
        self.auto_offset = True    # Converter works under auto_offset assumption
        #self.auto_file = True     # Redundant when working on file conversion, never used 
        
        # sampling rate
        self.fs = 1e3
        
        # transmit frequency
        self.f_transmit = 24.05e9

        # initialize buffers and output vectors
        self.init_buffers(buffer_len)
        
    # initialize buffers for data handling
    def init_buffers(self, buffer_len):
        
        # self.time_window = self.spin_internal_time.value() # Not used in converter. Hardcoded instead to buffer_len
        self.buffer_length = int(buffer_len)
        self.buffer = np.zeros((buffer_len, 6), dtype=np.int64)
        #self.i_buffer = np.zeros(self.buffer_length)
        #self.q_buffer = np.zeros(self.buffer_length)
        #self.ecg1_buffer = np.zeros(self.buffer_length)
        #self.ecg2_buffer = np.zeros(self.buffer_length)
        #self.ana1_buffer = np.zeros(self.buffer_length)
        #self.ana2_buffer = np.zeros(self.buffer_length)
        self.idata = np.zeros(self.buffer_length)
        self.qdata = np.zeros(self.buffer_length)
        self.idata_corrected = np.zeros(self.buffer_length)
        self.qdata_corrected = np.zeros(self.buffer_length)
        self.displacement = np.zeros(self.buffer_length)
        self.distance_breath = np.zeros(self.buffer_length)
        self.distance_pulse = np.zeros(self.buffer_length)
        self.distance_heartsound = np.zeros(self.buffer_length)
        self.hs1 = np.zeros(self.buffer_length)
        self.hs2 = np.zeros(self.buffer_length)
        self.ecg = np.zeros(self.buffer_length)

        # vitals continuity
        self.last_breath_logged = None
        self.last_pulse_logged = None
        self.last_heart_logged = None
        self.last_hs1_logged = None
        self.last_hs2_logged = None

    # get offset of current iq data buffers
    def get_offset(self):
        
        # select random popints for correction
        selected_points = np.random.choice(np.arange(0, self.buffer_length), 10)
        
        # estimate offset
        offset_points_i = self.buffer[:,0][selected_points]
        offset_points_q = self.buffer[:,1][selected_points]
        offset = estimate_offset(offset_points_i, offset_points_q)
        self.offset_i = offset[0] / 2 ** 24 * 4
        self.offset_q = offset[1] / 2 ** 24 * 4

        
class Filters:
    def __init__(self, fs, fc_ecg_us=False):
        order = 2

        # US config for ECG filtering
        fc_ecg_us = fc_ecg_us
        
        # Input values
        fc_breath_low, fc_breath_high = 0.05, 1 
        fc_pulse_low, fc_pulse_high = 0.66, 3 
        fc_heart_low, fc_heart_high = 15, 60 
        fc_ecg_main = 60 if fc_ecg_us else 50          # Europe config 50Hz, otherwise US config 60Hz
        fc_ecg_harmonic = 120 if fc_ecg_us else 100    # Europe config 100Hz, otherwise US config 120Hz
        fc_ecg_low = 40

        # bandpass filters
        self.breath_b, self.breath_a = butter(order, [fc_breath_low, fc_breath_high], fs=fs, btype='band')
        self.pulse_b,  self.pulse_a  = butter(order, [fc_pulse_low,  fc_pulse_high],  fs=fs, btype='band')
        self.heart_b,  self.heart_a  = butter(order, [fc_heart_low,  fc_heart_high],  fs=fs, btype='band')

        # ECG filters
        self.ecg_main_b, self.ecg_main_a  = iirnotch(fc_ecg_main, 5, fs=fs)
        self.ecg_harm_b, self.ecg_harm_a = iirnotch(fc_ecg_harmonic, 20, fs=fs)
        self.ecg_lowpass_b, self.ecg_lowpass_a = butter(order, fc_ecg_low, btype='lowpass', analog=False, fs=fs)
        
# main() creates the instances and executes conversion
def main():
    rf_file = ''
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the input file")
    args = parser.parse_args()
    
    rf_file = args.filename

    header_lines, ver, fmt = read_header(rf_file)
    if fmt =='' and ver == 1.2:
        rf_data = pd.read_csv(rf_file, comment="#", header=0)
    else:
        rf_data = pd.read_csv(rf_file, skiprows=4, names = [s.strip() for s in fmt.split(',')])
    fmt = ",".join(rf_data.columns)            
    
    state = StateRF()              # state created
    filt = Filters(fs=state.fs)    # filters created and initialized

    # Output file
    output_file = rf_file.split('.csv')[0]+"_converted.csv"

    #6-Buffer version
    # Step 1: Create output file
    with open(output_file, "w") as logfile:
        logfile.write("# Created with Sykno's ViRa24 Modified (converted val), UI version 1.2 \n")
        logfile.write(header_lines[1] + "\n") # Start time: %Y-%m-%d %H-%M-%S
        logfile.write("# Sample rate: " + str(int(state.fs)) + " Sa/s"+ "\n")
        logfile.write("# Format:\n" +
        "I_raw,Q_raw,ECG1_raw,ECG2_raw,ANA1_raw,ANA2_raw,"                                # Raw data
        "Distance_breath,Distance_pulse,Distance_heartsound,HS1_AC,HS2_AC,ECG_filtered" + # Processed data
        "\n")

    # Example source data 
    start_count = 200 if ver==1.2 else 0
    colnames = rf_data.columns

    # Load input file values and process as in 2-sample buffer 
    values = rf_data[colnames].values[start_count:,:6]   # Discard computed values when loading from file (redundant in v1.0)

    sample_cnt = 0
    # Process in chunks of 2, shifting up and appending at bottom
    for i in range(0, len(values), 2):
        update_val = values[i:i+2,:]
        k = update_val.shape[0]                     # handles the final chunk if it's length 1
        if k != 2:                                  # Avoid update if length 1 
            print(f'Last sample discarded: Odd number of samples k={k}, i,i+2=[{i},{i+1}]')
            break

        # Shift upwards by k and place new samples at the bottom end
        state.buffer[:-k,:] = state.buffer[k:,:] # roll deletion
        state.buffer[-k:,:] = update_val
        sample_cnt +=k

        # Use np.views for metrics and output values
        i_buffer = state.buffer[:,0]
        q_buffer = state.buffer[:,1]
        ecg1_buffer=state.buffer[:,2]
        ecg2_buffer=state.buffer[:,3]
        ana1_buffer = state.buffer[:,4]
        ana2_buffer = state.buffer[:,5]

        # trigger update 20 times per second (= every 50 samples @1kSps). No plot update
        framerate = 20
        frame_smaplecnt = int(state.fs/framerate)

        if np.mod(sample_cnt, frame_smaplecnt) == 0:

            state.idata = i_buffer / 2 ** 24 * 4  # stored in class
            state.qdata = q_buffer / 2 ** 24 * 4  # stored in class
            if not state.ac_coupling:
                #print(sample_cnt)
                state.idata_corrected = state.idata - state.offset_i
                state.qdata_corrected = state.qdata - state.offset_q

                # calculate iq phase angle
                phi = np.arctan2(state.qdata_corrected, state.idata_corrected)
                phi_linear = np.unwrap(phi)
                # calculate displacement
                state.displacement = (phi_linear * c) / (4 * np.pi * state.f_transmit) * 1e3

                state.distance_breath = filtfilt(filt.breath_b, filt.breath_a, state.displacement)
                state.distance_pulse = filtfilt(filt.pulse_b, filt.pulse_a, state.displacement)
                state.distance_heartsound = filtfilt(filt.heart_b, filt.heart_a, state.displacement)
                state.hs1 = np.nan*np.zeros(state.buffer_length)
                state.hs2 = np.nan*np.zeros(state.buffer_length)
                
            else:
                state.distance_breath = np.nan*np.zeros(state.buffer_length)
                state.distance_pulse = np.nan*np.zeros(state.buffer_length)
                state.distance_heartsound = np.nan*np.zeros(state.buffer_length)
                # update heart sounds
                state.hs1 = filtfilt(filt.heart_b, filt.heart_a, state.idata)
                state.hs2 = filtfilt(filt.heart_b, filt.heart_a, state.qdata)
                
            ecg = ecg2_buffer-ecg1_buffer
            ecg = filtfilt(filt.ecg_main_b, filt.ecg_main_a, ecg)
            ecg = filtfilt(filt.ecg_harm_b, filt.ecg_harm_a, ecg)
            ecg = filtfilt(filt.ecg_lowpass_b, filt.ecg_lowpass_a, ecg)
            ecg = ecg - np.mean(ecg)
            state.ecg = ecg / np.max(np.abs(ecg[200:-200]))

        # trigger logging and automatic offset estimation every 2.5 seconds
        offset_correction_period = 2.5
        offset_correction_framecnt = int(state.fs*offset_correction_period)
        if np.mod(sample_cnt, offset_correction_framecnt) == 0:
            # trigger logging every offset correction frame (last 2.5 seconds is logged)
            if state.logging:
                # Slice for the block we want to log (excludes last 200 samples) to avoid filter edge effects
                # E.g. from index `-2700` up to `-200`.
                logging_slice = slice(-(offset_correction_framecnt + 200), -200)

                # Identify the first new samples in this block (index 0 of logging_slice)
                breath_new_first = state.distance_breath[logging_slice][0]
                pulse_new_first = state.distance_pulse[logging_slice][0]
                heart_new_first = state.distance_heartsound[logging_slice][0]
                hs1_new_first = state.hs1[logging_slice][0]
                hs2_new_first = state.hs2[logging_slice][0]

                # Compute continuity shifts, if we have previously logged data
                if state.last_breath_logged is not None:
                    shift_breath = state.last_breath_logged - breath_new_first
                    shift_pulse = state.last_pulse_logged - pulse_new_first
                    shift_heartsound = state.last_heart_logged - heart_new_first
                    shift_hs1 = state.last_hs1_logged - hs1_new_first
                    shift_hs2 = state.last_hs2_logged - hs2_new_first

                else:
                    # If this is first time logging, no shift needed
                    shift_breath = shift_pulse = shift_heartsound = shift_hs1 = shift_hs2 = 0

                # Prepare the data array with needed shifts and scaling for heart
                data = np.column_stack([
                    i_buffer[logging_slice],
                    q_buffer[logging_slice],
                    ecg1_buffer[logging_slice],
                    ecg2_buffer[logging_slice],
                    ana1_buffer[logging_slice],
                    ana2_buffer[logging_slice],
                    state.distance_breath[logging_slice] + shift_breath,
                    state.distance_pulse[logging_slice] + shift_pulse,
                    state.distance_heartsound[logging_slice] * 1e3 + shift_heartsound,
                    state.hs1[logging_slice] + shift_hs1,
                    state.hs2[logging_slice] + shift_hs2,
                    state.ecg[logging_slice]
                ])

                # Save the data to CSV with full double precision
                with open(output_file, 'a') as logfile: 
                    np.savetxt(logfile, data, delimiter=",", fmt="%.17g")

                # Update internal state based on the final row in the newly logged block
                # (We don’t apply the 1e3 scaling here for heart)
                state.last_breath_logged = data[-1][6]
                state.last_pulse_logged = data[-1][7]
                state.last_heart_logged = state.distance_heartsound[-201] + shift_heartsound
                state.last_hs1_logged = data[-1][9]
                state.last_hs2_logged = data[-1][10]

            # do offset correction
            if state.auto_offset and sample_cnt > state.buffer_length and not state.ac_coupling:
                # select random points for correction
                selected_points = np.random.choice(np.arange(0, state.buffer_length), 10)
                # estimate offset
                offset_points_i = i_buffer[selected_points]
                offset_points_q = q_buffer[selected_points]
                offset = estimate_offset(offset_points_i, offset_points_q)
                offset_i_new = offset[0] / 2 ** 24 * 4
                offset_q_new = offset[1] / 2 ** 24 * 4

                if state.offset_i == 0:
                    # if offset was not set before
                    state.get_offset()
                else:
                    # calculate offset error and trigger offset update, if threshold eceeded
                    threshold = 0.2
                    deviation_i = np.abs((state.offset_i - offset_i_new)/state.offset_i)
                    deviation_q = np.abs((state.offset_q - offset_q_new)/state.offset_q)
                    if deviation_i > threshold or deviation_q > threshold:
                        state.get_offset()

    print("Conversions stored in ", output_file)

if __name__ == "__main__":
    main()
