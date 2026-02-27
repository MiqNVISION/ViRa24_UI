# ViRa24 Evaluation GUI
# Copyright (C) 2025 Sykno GmbH
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

# 3rd party user interface and display packages
import PyQt6
from PyQt6 import QtWidgets, uic, QtGui
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPalette, QColor
import pyqtgraph as pg

# 3rd party math packages
import numpy as np
from scipy.constants import c
from scipy.signal import butter, filtfilt, iirnotch

# 3rd party serial port interfacing
import serial
import serial.tools.list_ports

# standard library
import os
import sys
import time
import pathlib
from pathlib import Path
from threading import Thread
from datetime import datetime

# additional functions
from resources.vira_lib import estimate_offset


#%% signaling class
class SignalCommunicate(PyQt6.QtCore.QObject):
    request_graph_update = PyQt6.QtCore.pyqtSignal()
    msg_box = PyQt6.QtCore.pyqtSignal(str)


#%% main ui class
class MainWindow(QtWidgets.QMainWindow):

    # initialization
    def __init__(self):
        
        # software version
        self.ver = 1.2
        
        # sampling rate
        self.fs = 1e3
        
        # transmit frequency
        self.f_transmit = 24.05e9
        
        # set numpy error warning
        np.seterr(all='warn')
        
        # create ui
        super(MainWindow, self).__init__()
        uic.loadUi("resources/main_window.ui", self)
        self.label_version.setText("UI version " + str(self.ver))
        print("Sykno ViRa24 Evaluation GUI")
        print("UI version: " + str(self.ver))

        # init variables
        self.com_obj = []

        self.time_window = self.spin_internal_time.value()
        self.buffer_length = int(self.time_window * self.fs)
        self.ac_coupling = self.checkbox_ac.isChecked()
        
        self.offset_i = 0
        self.offset_q = 0
        self.auto_offset = True
        self.auto_file = True
                      
        self.logfile_handle = None
                
        # initialization functions
        self.init_buffers()
        self.init_filter()
        self.init_plots()
        self.set_internal_time()
        
        # initialize signals
        self.signalComm = SignalCommunicate()
        self.signalComm.request_graph_update.connect(self.update_graph)
        self.signalComm.msg_box.connect(self.msg_box)   
        
        # refresh list of com ports
        self.refresh_comports()
        
        # initialize logging
        self.save_dir = pathlib.Path().absolute()
        self.script_dir = pathlib.Path().absolute()
        self.save_dir = Path.joinpath(self.script_dir, "logs")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.logging = False
        self.running = False
        
        
    # initialize buffers for data handling
    def init_buffers(self):
        
        self.time_window = self.spin_internal_time.value()
        self.buffer_length = int(self.time_window * self.fs)
        self.i_buffer = np.zeros(self.buffer_length)
        self.q_buffer = np.zeros(self.buffer_length)
        self.ecg1_buffer = np.zeros(self.buffer_length)
        self.ecg2_buffer = np.zeros(self.buffer_length)
        self.ana1_buffer = np.zeros(self.buffer_length)
        self.ana2_buffer = np.zeros(self.buffer_length)
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
        self.last_breath_logged = None
        self.last_pulse_logged = None
        self.last_heart_logged = None
        self.last_hs1_logged = None
        self.last_hs2_logged = None

    # initialize filters for data processing
    def init_filter(self):
        
        # cutoff frequencies
        fc_breath_low = self.spin_fc_min_breath.value()
        fc_pulse_low = self.spin_fc_min_pulse.value()
        fc_heart_low = self.spin_fc_min_heart.value()
        fc_breath_high = self.spin_fc_max_breath.value()
        fc_pulse_high = self.spin_fc_max_pulse.value()
        fc_heart_high = self.spin_fc_max_heart.value()
        fc_ecg_main = 60 if self.radio_fc_ecg_us.isChecked() else 50
        fc_ecg_harmonic = 120 if self.radio_fc_ecg_us.isChecked() else 100

        # filter order
        order = 2

        # vital sign filter coefficients
        self.filter_breath_b, self.filter_breath_a = butter(order, [fc_breath_low, fc_breath_high], btype='bandpass', analog=False, output='ba', fs=self.fs)
        self.filter_pulse_b, self.filter_pulse_a = butter(order, [fc_pulse_low, fc_pulse_high], btype='bandpass', analog=False, output='ba', fs=self.fs)
        self.filter_heart_b, self.filter_heart_a = butter(order, [fc_heart_low, fc_heart_high], btype='bandpass', analog=False, output='ba', fs=self.fs)
        
        # ecg filter coefficients
        # 50 Hz notch filter
        self.filter1_ecg_b, self.filter1_ecg_a = iirnotch(w0=fc_ecg_main, Q=5, fs=self.fs)
        # 100 Hz notch filter
        self.filter2_ecg_b, self.filter2_ecg_a = iirnotch(w0=fc_ecg_harmonic, Q=20, fs=self.fs)
        # ECG lowpass filter
        self.filter3_ecg_b, self.filter3_ecg_a = butter(order, 40, btype='lowpass', analog=False, fs=self.fs)    
        
        
     # initialize plots
    def init_plots(self):
        
        # plot i and q over time
        self.plot_iq_time.addLegend()
        self.plotline_iq_i = pg.PlotDataItem(name='I raw', pen=pg.mkPen(color=(0, 114, 189)))
        self.plotline_iq_q = pg.PlotDataItem(name='Q raw', pen=pg.mkPen(color=(162, 20, 29)))
        self.plot_iq_time.addItem(self.plotline_iq_i)
        self.plot_iq_time.addItem(self.plotline_iq_q)
        self.plot_iq_time.setTitle("Raw data I/Q")
        self.plot_iq_time.setLabel('bottom', 'Time in s')
        self.plot_iq_time.setLabel('left', 'Voltage in V')
        self.plot_iq_time.setXRange(-self.time_window, 0)
        self.plot_iq_time.setYRange(-2.1, 2.1)
        self.set_plotstyle(self.plot_iq_time)
        # clipping zones
        self.plot_iq_time_upperlimit1 = pg.PlotDataItem()
        self.plot_iq_time_upperlimit2 = pg.PlotDataItem()
        self.plot_iq_time_lowerlimit1 = pg.PlotDataItem()
        self.plot_iq_time_lowerlimit2 = pg.PlotDataItem()
        self.plot_iq_time_upperlimitfill = pg.FillBetweenItem(self.plot_iq_time_upperlimit1, self.plot_iq_time_upperlimit2, brush=(255,0,0,100))
        self.plot_iq_time_lowerlimitfill = pg.FillBetweenItem(self.plot_iq_time_lowerlimit1, self.plot_iq_time_lowerlimit2, brush=(255,0,0,100))
        self.plot_iq_time_upperlimit1.setData([-10,10],[1.6, 1.6])
        self.plot_iq_time_upperlimit2.setData([-10,10],[4, 4])
        self.plot_iq_time_lowerlimit1.setData([-10,10],[-1.6, -1.6])
        self.plot_iq_time_lowerlimit2.setData([-10,10],[-4, -4])
        self.plot_iq_time.addItem(self.plot_iq_time_upperlimitfill, ignoreBounds = True)
        self.plot_iq_time.addItem(self.plot_iq_time_lowerlimitfill, ignoreBounds = True)
        self.plot_iq_time.enableAutoRange()

        # plot ellipse (i over q)
        # cross
        self.plot_iq_ellipse_inflineI = pg.InfiniteLine(0,90, (255, 255, 255))
        self.plot_iq_ellipse_inflineQ = pg.InfiniteLine(0,0, (255, 255, 255))
        self.plot_iq_ellipse.addItem(self.plot_iq_ellipse_inflineI, ignoreBounds = True)
        self.plot_iq_ellipse.addItem(self.plot_iq_ellipse_inflineQ, ignoreBounds = True)
        # data
        self.plotline_ellipse = pg.PlotDataItem(pen=pg.mkPen(color=(0, 114, 189)))
        self.offset_points = pg.ScatterPlotItem(name='Offset', brush=(162, 20, 29), pen=None, size=10, symbol='x')
        self.plot_iq_ellipse.addItem(self.plotline_ellipse)
        self.plot_iq_ellipse.addItem(self.offset_points)
        self.plot_iq_ellipse.setTitle("I/Q diagram")
        self.plot_iq_ellipse.setLabel("bottom", "Inphase voltage in V")
        self.plot_iq_ellipse.setLabel("left", "Quadrature voltage in V")
        self.plot_iq_ellipse.setXRange(-1.6, 1.6)
        self.plot_iq_ellipse.setYRange(-1.6, 1.6)
        self.plot_iq_ellipse.setAspectLocked()
        self.plot_iq_ellipse.enableAutoRange()
        self.set_plotstyle(self.plot_iq_ellipse)
        # clipping zones
        self.plot_iq_ellipse_upperlimit1 = pg.PlotDataItem()
        self.plot_iq_ellipse_upperlimit2 = pg.PlotDataItem()
        self.plot_iq_ellipse_lowerlimit1 = pg.PlotDataItem()
        self.plot_iq_ellipse_lowerlimit2 = pg.PlotDataItem()
        self.plot_iq_ellipse_leftlimit1 = pg.PlotDataItem()
        self.plot_iq_ellipse_leftlimit2 = pg.PlotDataItem()
        self.plot_iq_ellipse_rightlimit1 = pg.PlotDataItem()
        self.plot_iq_ellipse_rightlimit2 = pg.PlotDataItem()
        self.plot_iq_ellipse_upperlimitfill = pg.FillBetweenItem(self.plot_iq_ellipse_upperlimit1, self.plot_iq_ellipse_upperlimit2, brush=(255,0,0,100))
        self.plot_iq_ellipse_lowerlimitfill = pg.FillBetweenItem(self.plot_iq_ellipse_lowerlimit1, self.plot_iq_ellipse_lowerlimit2, brush=(255,0,0,100))
        self.plot_iq_ellipse_leftlimitfill = pg.FillBetweenItem(self.plot_iq_ellipse_leftlimit1, self.plot_iq_ellipse_leftlimit2, brush=(255,0,0,100))
        self.plot_iq_ellipse_rightlimitfill = pg.FillBetweenItem(self.plot_iq_ellipse_rightlimit1, self.plot_iq_ellipse_rightlimit2, brush=(255,0,0,100))
        self.plot_iq_ellipse_upperlimit1.setData([-10,10],[1.6, 1.6])
        self.plot_iq_ellipse_upperlimit2.setData([-10,10],[4, 4])
        self.plot_iq_ellipse_lowerlimit1.setData([-10,10],[-1.6, -1.6])
        self.plot_iq_ellipse_lowerlimit2.setData([-10,10],[-4, -4])
        self.plot_iq_ellipse_leftlimit1.setData([-10,-10],[-1.6, 1.6])
        self.plot_iq_ellipse_leftlimit2.setData([-1.6,-1.6],[-1.6, 1.6])
        self.plot_iq_ellipse_rightlimit1.setData([10,10],[-1.6, 1.6])
        self.plot_iq_ellipse_rightlimit2.setData([1.6,1.6],[-1.6, 1.6])
        self.plot_iq_ellipse.addItem(self.plot_iq_ellipse_upperlimitfill, ignoreBounds = True)
        self.plot_iq_ellipse.addItem(self.plot_iq_ellipse_lowerlimitfill, ignoreBounds = True)
        self.plot_iq_ellipse.addItem(self.plot_iq_ellipse_leftlimitfill, ignoreBounds = True)
        self.plot_iq_ellipse.addItem(self.plot_iq_ellipse_rightlimitfill, ignoreBounds = True)

        # plot distance over time
        self.plotline_displacement = pg.PlotDataItem(name='I raw', pen=pg.mkPen(color=(0, 114, 189)))
        self.plot_displacement.addItem(self.plotline_displacement)
        self.plot_displacement.setTitle("Displacement (unfiltered)")
        self.plot_displacement.setLabel("bottom", "Time in s")
        self.plot_displacement.setLabel("left", "Displacement in mm")
        self.set_plotstyle(self.plot_displacement)

        # plot spectrum
        self.plot_spectrum.addLegend()
        self.plotline_spectrum_i = pg.PlotDataItem(name='Inphase spectrum', pen=pg.mkPen(color=(0, 114, 189)))
        self.plotline_spectrum_q = pg.PlotDataItem(name='Quadrature spectrum', pen=pg.mkPen(color=(162, 20, 29)))
        self.plot_spectrum.addItem(self.plotline_spectrum_i)
        self.plot_spectrum.addItem(self.plotline_spectrum_q)
        self.plot_spectrum.setTitle("Raw data frequency spectrum")
        self.plot_spectrum.setLabel("bottom", "Frequency in Hz")
        self.plot_spectrum.setLabel("left", "Magnitude")
        self.plot_spectrum.setLogMode(x=False, y=True)
        self.plot_spectrum.setXRange(10, 100)
        self.plot_spectrum.setYRange(-3, 2)
        self.set_plotstyle(self.plot_spectrum)

        # plot breath signal
        self.plotline_breath = pg.PlotDataItem(name='breath', pen=pg.mkPen(color=(0, 114, 189)))
        self.plot_breath.addItem(self.plotline_breath)
        self.plot_breath.setTitle("Displacement")
        self.plot_breath.setLabel("bottom", "Time in s")
        self.plot_breath.setLabel("left", "Displacement in mm")
        self.plot_breath.setYRange(-5, 5)
        self.set_plotstyle(self.plot_breath)

        # plot pulse signal
        self.plotline_pulse = pg.PlotDataItem(name='pulse', pen=pg.mkPen(color=(0, 114, 189)))
        self.plot_pulse.addItem(self.plotline_pulse)
        self.plot_pulse.setTitle("Displacement")
        self.plot_pulse.setLabel("bottom", "Time in s")
        self.plot_pulse.setLabel("left", "Displacement in mm")
        self.set_plotstyle(self.plot_pulse)

        # plot heart sound signal
        self.plotline_heartsounds = pg.PlotDataItem(pen=pg.mkPen(color=(0, 114, 189)))
        self.plotline_heartsounds_ac = pg.PlotDataItem(pen=pg.mkPen(color=(162, 20, 29)))
        self.plot_heartsounds.addItem(self.plotline_heartsounds)
        self.plot_heartsounds.addItem(self.plotline_heartsounds_ac)
        self.plot_heartsounds.setTitle("Displacement")
        self.plot_heartsounds.setLabel("bottom", "Time in s")
        self.plot_heartsounds.setLabel("left", "Displacement in µm")
        self.set_plotstyle(self.plot_heartsounds)
 
        # plot ecg signal
        self.plotline_ecg = pg.PlotDataItem(pen=pg.mkPen(color=(0, 114, 189)))
        self.plot_ecg.addItem(self.plotline_ecg)
        self.plot_ecg.setTitle("ECG")
        self.plot_ecg.setLabel("bottom", "Time in s")
        self.plot_ecg.setLabel("left", "ECG (arb. units)")
        self.plot_ecg.setYRange(-1.1, 1.1)
        self.set_plotstyle(self.plot_ecg)


    # set the style of the plots
    def set_plotstyle(self, plot):
        
        # curves to be drawn on the graph
        axisfont = QtGui.QFont()
        axisfont.setPixelSize(14)
        dark_text_color = QColor(200, 200, 200)
        plot.getAxis('left').setPen(dark_text_color)
        plot.getAxis('left').setTextPen(dark_text_color)
        plot.getAxis("left").setTickFont(axisfont)
        plot.getAxis("left").setStyle(tickTextOffset=10)
        plot.getAxis('left').label.setFont(axisfont)
        plot.getAxis('bottom').setPen(dark_text_color)
        plot.getAxis('bottom').setTextPen(dark_text_color)
        plot.getAxis("bottom").setTickFont(axisfont)
        plot.getAxis("bottom").setStyle(tickTextOffset=10)
        plot.getAxis('bottom').label.setFont(axisfont)

  
    # create a message box for warnings   
    def msg_box(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle('Warning')
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    
    # send data via com port
    def com_send(self, cmd):
        try:
            self.com_obj.write(bytes(cmd, "utf-8"))
        except:
            pass
 
    
    # refresh list of available com ports
    def refresh_comports(self):
        self.list_ports.clear()
        ports = serial.tools.list_ports.comports()           
        if not ports:
            self.button_startstop.setEnabled(False)
        else:
            self.button_startstop.setEnabled(True)
            for p in ports:
                self.list_ports.addItem(p.device)
    
    
    # callback for  start/stop measurement button
    # - activates transmit power and data streaming
    # - sets amplification and coupling according to ui values
    # - creates log file if requested
    # - starts thread for data reception and processing
    def start_stop(self):
        if not self.running:
            # system stopped -> start system
            print("Start streaming...")
            self.running = True
            # start system and set parameters
            self.com_send("#start#")
            time.sleep(0.2)
            self.set_amps_i()
            time.sleep(0.1)
            self.set_amps_q()
            time.sleep(0.1)
            self.set_coupling()
            
            self.button_startstop.setText("Stop")
            self.checkbox_log.setEnabled(False)
            self.combobox_filename.setEnabled(False)
            self.button_connect.setEnabled(False)
            
            # create and open log file
            if self.logging:
                if self.auto_file:
                    # auto file name
                    self.save_file = Path.joinpath(self.save_dir, (datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"))
                else:
                    # manual file name
                    if not self.save_file:
                        self.signalComm.msg_box.emit("Please choose a log file name.")
                        self.start_stop()
                        return
                   
                self.logfile_handle = open(self.save_file, 'a+')
                self.lineedit_filename.setText(str(self.save_file))

                self.logfile_handle.write("# Created with Sykno's ViRa24, UI version " + str(self.ver) + "\n")
                self.logfile_handle.write("# Start time: " + datetime.now().strftime("%Y-%m-%d %H-%M-%S")+ "\n")
                self.logfile_handle.write("# Sample rate: " + str(int(self.fs)) + " Sa/s"+ "\n")
                self.logfile_handle.write("# Format:\n" +
                                          "I_raw,Q_raw,ECG1_raw,ECG2_raw,ANA1_raw,ANA2_raw,"                                # Raw data
                                          "Distance_breath,Distance_pulse,Distance_heartsound,HS1_AC,HS2_AC,ECG_filtered" + # Processed data
                                          "\n")
            
            # start thread for data reception
            self.com_thread = Thread(target=self.com_read)
            self.com_thread.start()
            
        else:
            # system running -> stop system
            print("Stop streaming...")
            self.running = False
            # stop system 
            self.com_send("#stop#")
            
            self.button_startstop.setText("Start")
            self.checkbox_log.setEnabled(True)
            self.combobox_filename.setEnabled(True)
            self.button_connect.setEnabled(True)

            # close logfile
            if self.logging and self.logfile_handle:
                self.logfile_handle.close()
                self.lineedit_filename.setText("")
                self.save_file = None


    # read data from com port
    # - receive data continuously
    # - decompose into adc samples
    # - write to buffers
    # - trigger graph updates
    # - trigger automatic offset estimation
    def com_read(self):
        sample_cnt = 0
        raw_buffer = b''

        # loop forever for contiuous data collection
        while self.running:
            try:
                new_data = self.com_obj.read(self.com_obj.inWaiting() or 1)               
                # if no data could be read
                if len(new_data) == 0:
                    if self.running:
                        # data acquisition should be running -> exception
                        raise(Exception())
                    else:
                        # data acquisition should be terminate -> stop loop
                        break
                else:
                    # append to buffer
                    raw_buffer += new_data
            except Exception as e:
                print(e)
                self.signalComm.msg_box.emit("COM port timeout.\nDid you choose the right COM port?")
                self.set_ui_connected(False)
                self.start_stop()

            # process received data buffer
            while b'#DATA#' in raw_buffer:
                raw_frame, raw_buffer = raw_buffer.split(b'#DATA#', 1)
                decoded_frame = np.frombuffer(raw_frame, np.uint8)
                
                # if received frame has wrong length
                if len(decoded_frame) != 48:
                    pass
                
                # reconstruct adc samples (two samples per channel per frame)
                val00 = int.from_bytes(decoded_frame[0:3], byteorder='little', signed=True)
                val10 = int.from_bytes(decoded_frame[4:7], byteorder='little', signed=True)
                val20 = int.from_bytes(decoded_frame[8:11], byteorder='little', signed=True)
                val30 = int.from_bytes(decoded_frame[12:15], byteorder='little', signed=True)
                val40 = int.from_bytes(decoded_frame[16:19], byteorder='little', signed=True)
                val50 = int.from_bytes(decoded_frame[20:23], byteorder='little', signed=True)
                
                val01 = int.from_bytes(decoded_frame[24:27], byteorder='little', signed=True)
                val11 = int.from_bytes(decoded_frame[28:31], byteorder='little', signed=True)
                val21 = int.from_bytes(decoded_frame[32:35], byteorder='little', signed=True)
                val31 = int.from_bytes(decoded_frame[36:39], byteorder='little', signed=True)
                val41 = int.from_bytes(decoded_frame[40:43], byteorder='little', signed=True)
                val51 = int.from_bytes(decoded_frame[44:47], byteorder='little', signed=True)
                   
                # write buffers
                self.i_buffer = np.concatenate((self.i_buffer[2:], [val00, val01]))  # inphase channel
                self.q_buffer = np.concatenate((self.q_buffer[2:], [val10, val11]))  # quadrature channel
                self.ecg1_buffer = np.concatenate((self.ecg1_buffer[2:], [val20, val21]))  # ECG channel 1
                self.ecg2_buffer = np.concatenate((self.ecg2_buffer[2:], [val30, val31]))  # ECG channel 2
                self.ana1_buffer = np.concatenate((self.ana1_buffer[2:], [val40, val41]))  # analog input channel 1
                self.ana2_buffer = np.concatenate((self.ana2_buffer[2:], [val50, val51]))  # analog input channel 2
                                 
                sample_cnt += 2
                           
                # trigger plot update 20 times per second (= every 50 samples @1kSps)
                framerate = 20
                frame_smaplecnt = int(self.fs/framerate)
                                
                if np.mod(sample_cnt, frame_smaplecnt) == 0:

                    self.idata = self.i_buffer / 2 ** 24 * 4
                    self.qdata = self.q_buffer / 2 ** 24 * 4
                    if not self.ac_coupling:
                        self.idata_corrected = self.idata - self.offset_i
                        self.qdata_corrected = self.qdata - self.offset_q
                        # calculate iq phase angle
                        phi = np.arctan2(self.qdata_corrected, self.idata_corrected)
                        phi_linear = np.unwrap(phi)
                        # calculate displacement
                        self.displacement = (phi_linear * c) / (4 * np.pi * self.f_transmit) * 1e3

                        self.distance_breath = filtfilt(self.filter_breath_b, self.filter_breath_a, self.displacement)
                        self.distance_pulse = filtfilt(self.filter_pulse_b, self.filter_pulse_a, self.displacement)
                        self.distance_heartsound = filtfilt(self.filter_heart_b, self.filter_heart_a, self.displacement)
                        self.hs1 = np.nan*np.zeros(self.buffer_length)
                        self.hs2 = np.nan*np.zeros(self.buffer_length)
                        ecg = self.ecg2_buffer - self.ecg1_buffer
                        ecg = filtfilt(self.filter1_ecg_b, self.filter1_ecg_a, ecg)
                        ecg = filtfilt(self.filter2_ecg_b, self.filter2_ecg_a, ecg)
                        ecg = filtfilt(self.filter3_ecg_b, self.filter3_ecg_a, ecg)
                        ecg = ecg - np.mean(ecg)
                        self.ecg = ecg / np.max(np.abs(ecg[200:-200]))
                    else:
                        self.distance_breath = np.nan*np.zeros(self.buffer_length)
                        self.distance_pulse = np.nan*np.zeros(self.buffer_length)
                        self.distance_heartsound = np.nan*np.zeros(self.buffer_length)
                        # update heart sounds
                        self.hs1 = filtfilt(self.filter_heart_b, self.filter_heart_a, self.idata)
                        self.hs2 = filtfilt(self.filter_heart_b, self.filter_heart_a, self.qdata)
                        ecg = self.ecg2_buffer - self.ecg1_buffer
                        ecg = filtfilt(self.filter1_ecg_b, self.filter1_ecg_a, ecg)
                        ecg = filtfilt(self.filter2_ecg_b, self.filter2_ecg_a, ecg)
                        ecg = filtfilt(self.filter3_ecg_b, self.filter3_ecg_a, ecg)
                        ecg = ecg - np.mean(ecg)
                        ecg = ecg / np.max(np.abs(ecg[200:-200]))
                        self.ecg = ecg
                    # trigger plot updates
                    self.signalComm.request_graph_update.emit()   

                # trigger logging and automatic offset estimation every 2.5 seconds
                offset_correction_period = 2.5
                offset_correction_framecnt = int(self.fs*offset_correction_period)
                if np.mod(sample_cnt, offset_correction_framecnt) == 0:
                    # trigger logging every offset correction frame (last 2.5 seconds is logged)
                    if self.logging:
                        # Slice for the block we want to log (excludes last 200 samples) to avoid filter edge effects
                        # E.g. from index `-2700` up to `-200`.
                        logging_slice = slice(-(offset_correction_framecnt + 200), -200)

                        # Identify the first new samples in this block (index 0 of logging_slice)
                        breath_new_first = self.distance_breath[logging_slice][0]
                        pulse_new_first = self.distance_pulse[logging_slice][0]
                        heart_new_first = self.distance_heartsound[logging_slice][0]
                        hs1_new_first = self.hs1[logging_slice][0]
                        hs2_new_first = self.hs2[logging_slice][0]

                        # Compute continuity shifts, if we have previously logged data
                        if self.last_breath_logged is not None:
                            shift_breath = self.last_breath_logged - breath_new_first
                            shift_pulse = self.last_pulse_logged - pulse_new_first
                            shift_heartsound = self.last_heart_logged - heart_new_first
                            shift_hs1 = self.last_hs1_logged - hs1_new_first
                            shift_hs2 = self.last_hs2_logged - hs2_new_first
                        else:
                            # If this is first time logging, no shift needed
                            shift_breath = shift_pulse = shift_heartsound = shift_hs1 = shift_hs2 = 0

                        # Prepare the data array with needed shifts and scaling for heart
                        data = np.column_stack([
                            self.i_buffer[logging_slice],
                            self.q_buffer[logging_slice],
                            self.ecg1_buffer[logging_slice],
                            self.ecg2_buffer[logging_slice],
                            self.ana1_buffer[logging_slice],
                            self.ana2_buffer[logging_slice],
                            self.distance_breath[logging_slice] + shift_breath,
                            self.distance_pulse[logging_slice] + shift_pulse,
                            self.distance_heartsound[logging_slice] * 1e3 + shift_heartsound * 1e3,
                            self.hs1[logging_slice] + shift_hs1,
                            self.hs2[logging_slice] + shift_hs2,
                            self.ecg[logging_slice],
                        ])

                        # Save the data to CSV with full double precision
                        np.savetxt(self.logfile_handle, data, delimiter=",", fmt="%.17g")

                        # Update internal state based on the final row in the newly logged block
                        # (We don’t apply the 1e3 scaling here for heart)
                        self.last_breath_logged = data[-1][6]
                        self.last_pulse_logged = data[-1][7]
                        self.last_heart_logged = self.distance_heartsound[-201] + shift_heartsound
                        self.last_hs1_logged = data[-1][9]
                        self.last_hs2_logged = data[-1][10]

                    # do offset correction
                    if self.auto_offset and sample_cnt > self.buffer_length and not self.ac_coupling:
                        # select random points for correction
                        selected_points = np.random.choice(np.arange(0, self.buffer_length), 10)
                        # estimate offset
                        offset_points_i = self.i_buffer[selected_points]
                        offset_points_q = self.q_buffer[selected_points]
                        offset = estimate_offset(offset_points_i, offset_points_q)
                        offset_i_new = offset[0] / 2 ** 24 * 4
                        offset_q_new = offset[1] / 2 ** 24 * 4
                        
                        if self.offset_i == 0:
                            # if offset was not set before
                            self.get_offset()
                        else:
                            # calculate offset error and trigger offset update, if threshold eceeded
                            threshold = 0.2
                            deviation_i = np.abs((self.offset_i - offset_i_new)/self.offset_i)
                            deviation_q = np.abs((self.offset_q - offset_q_new)/self.offset_q)
                            if deviation_i > threshold or deviation_q > threshold:
                                self.get_offset()


    # update graphs
    def update_graph(self):
        
        time_axis = np.linspace(-self.time_window, 0, self.buffer_length)

        # if dc coupling is active
        if not self.ac_coupling:
            # update displacement plot
            if self.identify_tab() == 0:
                self.plotline_displacement.setData(time_axis, self.displacement)
            # update spectrum plot
            if self.identify_tab() == 1:
                f_axis = np.fft.rfftfreq(n=self.buffer_length, d=1/self.fs)
                spectrum_i = np.abs(np.fft.rfft(self.idata-np.mean(self.idata)))/self.buffer_length *1e3
                spectrum_q = np.abs(np.fft.rfft(self.qdata-np.mean(self.qdata)))/self.buffer_length *1e3
                self.plotline_spectrum_i.setData(f_axis, spectrum_i)
                self.plotline_spectrum_q.setData(f_axis, spectrum_q)
            # update vital sign plots
            if self.identify_tab() == 2:
                self.plotline_breath.setData(time_axis, self.distance_breath)
                self.plotline_pulse.setData(time_axis, self.distance_pulse)
                self.plotline_heartsounds.setData(time_axis, self.distance_heartsound*1e3)
                # plot ecg signal
                self.plotline_ecg.setData(time_axis[200:-200], self.ecg[200:-200])
            # update iq plots
            if self.identify_tab() == 3:
                # update i and q over time
                self.plotline_iq_i.setData(time_axis, self.idata)
                self.plotline_iq_q.setData(time_axis, self.qdata)
                # plot ellipse
                self.plotline_ellipse.setData(self.idata_corrected, self.qdata_corrected)

        # if ac coupling is active
        else:
            # update spectrum plot
            if self.identify_tab() == 1:
                f_axis = np.fft.rfftfreq(n=self.buffer_length, d=1/self.fs)
                spectrum_i = np.abs(np.fft.rfft(self.idata-np.mean(self.idata)))/self.buffer_length *1e3
                spectrum_q = np.abs(np.fft.rfft(self.qdata-np.mean(self.qdata)))/self.buffer_length *1e3
                self.plotline_spectrum_i.setData(f_axis, spectrum_i)
                self.plotline_spectrum_q.setData(f_axis, spectrum_q)
            if self.identify_tab() == 2:
                # update heart sounds
                self.plotline_heartsounds.setData(time_axis, self.hs1)
                self.plotline_heartsounds_ac.setData(time_axis, self.hs2)
                # plot ecg signal
                self.plotline_ecg.setData(time_axis[200:-200], self.ecg[200:-200])
            # update iq plots
            if self.identify_tab() == 3:
                # update i and q over time
                self.plotline_iq_i.setData(time_axis, self.idata)
                self.plotline_iq_q.setData(time_axis, self.qdata)
                # plot ellipse
                self.plotline_ellipse.setData(self.idata, self.qdata)

    # get offset of current iq data buffers
    def get_offset(self):
        
        # select random popints for correction
        selected_points = np.random.choice(np.arange(0, self.buffer_length), 10)
        
        # estimate offset
        offset_points_i = self.i_buffer[selected_points]
        offset_points_q = self.q_buffer[selected_points]
        offset = estimate_offset(offset_points_i, offset_points_q)
        self.offset_i = offset[0] / 2 ** 24 * 4
        self.offset_q = offset[1] / 2 ** 24 * 4
         
        # draw used points for offset estimation iq diagram
        self.offset_points.setData(offset_points_i / 2 ** 24 * 4 - self.offset_i, offset_points_q / 2 ** 24 * 4 - self.offset_q)
       
        # adjust saturation limits in iq diagram
        self.plot_iq_ellipse_upperlimit1.setData([-10,10]-self.offset_i,[1.6, 1.6]-self.offset_q)
        self.plot_iq_ellipse_upperlimit2.setData([-10,10]-self.offset_i,[4, 4]-self.offset_q)
        self.plot_iq_ellipse_lowerlimit1.setData([-10,10]-self.offset_i,[-1.6, -1.6]-self.offset_q)
        self.plot_iq_ellipse_lowerlimit2.setData([-10,10]-self.offset_i,[-4, -4]-self.offset_q)
        self.plot_iq_ellipse_leftlimit1.setData([-10,-10]-self.offset_i,[-1.6, 1.6]-self.offset_q)
        self.plot_iq_ellipse_leftlimit2.setData([-1.6,-1.6]-self.offset_i,[-1.6, 1.6]-self.offset_q)
        self.plot_iq_ellipse_rightlimit1.setData([10,10]-self.offset_i,[-1.6, 1.6]-self.offset_q)
        self.plot_iq_ellipse_rightlimit2.setData([1.6,1.6]-self.offset_i,[-1.6, 1.6]-self.offset_q)
    
    
    # callback function for logging checkbox
    def set_logging(self, state):
        
        self.logging = state
        if state:
            self.combobox_filename.setEnabled(True)
        else:
            self.combobox_filename.setEnabled(False)
     
        
    # callback function for logging file name dropdown
    def set_filename(self, index):

        if index == 1:
            # manual file name
            self.auto_file = False

            new_save_file = QFileDialog.getSaveFileName(self, "Select file to save...", str(self.save_dir), "Comma-separated value file (*.csv)")[0]
            
            if new_save_file:
                self.save_file = new_save_file
                print("filename:" + self.save_file)
                self.lineedit_filename.setText(self.save_file)
            
            else: 
                self.lineedit_filename.setText("No file selected")
                self.save_file = None         
        else: 
            # auto file name
            self.auto_file = True


    # callback function to set amplification of inphase signal
    def set_amps_i(self):
        amp_i = self.spin_gain_i.value()
        self.com_send("#gain#1#" + str(amp_i) +"#")


    # callback function to set amplification of quadrature signal
    def set_amps_q(self):
        amp_q = self.spin_gain_q.value()
        self.com_send("#gain#0#" + str(amp_q) +"#")


    # callback function to set coupling state
    def set_coupling(self):
        if self.running:
            # ac coupling
            if self.checkbox_ac.isChecked():
                self.com_send("#coupl#1#")
                self.ac_coupling = True
                self.init_buffers()
                
                self.plotline_displacement.setVisible(False)
                self.plotline_breath.setVisible(False)
                self.plotline_pulse.setVisible(False)
                self.offset_points.setVisible(False)
                self.plotline_heartsounds_ac.setVisible(True)
                self.plot_heartsounds.setLabel("left", "Displacement (arbitrary units)")
    
            # dccoupling
            else:
                self.com_send("#coupl#0#")
                self.ac_coupling = False
                self.init_buffers()
                self.plotline_displacement.setVisible(True)
                self.plotline_breath.setVisible(True)
                self.plotline_pulse.setVisible(True)
                self.offset_points.setVisible(True)
                self.plotline_heartsounds_ac.setVisible(False)
                self.plot_heartsounds.setLabel("left", "Displacement in µm")
     
        
    # open com port
    def set_connect(self):
        
        if not self.com_obj:
            # disconnected -> connect
            com_port = self.list_ports.currentText()
            
            if com_port:
                # if port is not empty
                try:
                    self.com_obj = serial.Serial(com_port, baudrate=115200, timeout=1)
        
                except:
                    print('Could not open port: ' + com_port)
                    self.com_obj = None
                    self.signalComm.msg_box.emit("Could not open COM port.\nIs it blocked by another application?")   
                    return
            else:
                # no com port selected
                return
        
            # connect was successful
            self.set_ui_connected(True)
 
        else:
            # connected -> disconnect
            self.com_obj.close()
            self.com_obj = None
            self.set_ui_connected(False)


    # callback function to open com port
    def set_ui_connected(self, state):
        if state:
            self.button_connect.setText("Disconnect")
            
        else:
            self.button_connect.setText("Connect")
        
        self.group_measurement.setEnabled(state)
        self.group_offset.setEnabled(state)
        self.group_baseband.setEnabled(state)
        self.group_timebasis.setEnabled(state)
        self.group_advanced.setEnabled(state)
     
        
    # callback function to send raw command
    def set_rawcmd(self):
        cmd = self.edit_rawcmd.text()
        self.com_send(cmd)


    # callback function to set automatic opffset compensation
    def set_auto_offset(self):
        self.auto_offset = self.checkbox_auto_offset.isChecked()


    # callback function when tab in ui was switched
    def identify_tab(self):
        return self.tabWidget.currentIndex()
    
    
    # callback function to start update mode
    def set_update(self):
        cmd = "#update#"
        self.com_send(cmd)
     
        
    # callback function to change filter parameters
    def set_filters(self):
        min_spins = [self.spin_fc_min_breath, self.spin_fc_min_pulse, self.spin_fc_min_heart]
        max_spins = [self.spin_fc_max_breath, self.spin_fc_max_pulse, self.spin_fc_max_heart]
        
        valid_config = True
        for k in range(3):
            # invalid filter settings
            if not min_spins[k].value() < max_spins[k].value():
                min_spins[k].setStyleSheet("color: red;")
                max_spins[k].setStyleSheet("color: red;")
                valid_config = False
            # valid filter settings                
            else:
                min_spins[k].setPalette(self.palette())
                min_spins[k].setStyleSheet("")
                max_spins[k].setPalette(self.palette())
                max_spins[k].setStyleSheet("")
                
        if valid_config:
            self.init_filter()


    # callback function to change time axis
    def set_internal_time(self):
        self.init_buffers()
        self.plot_displacement.setXRange(-self.time_window, 0)
        self.plot_breath.setXRange(-self.time_window, 0)
        self.plot_pulse.setXRange(-self.time_window, 0)
        self.plot_heartsounds.setXRange(-self.time_window, 0)
        self.plot_ecg.setXRange(-self.time_window, 0)


    # callback function to exit program
    def closeEvent(self, event):
        if self.running:
            self.start_stop()
        if self.com_obj:
            self.com_obj.close()
        time.sleep(0.2)


#%% main function
if __name__ == '__main__':
    
    # for OpenGL rendering only:
    # pg.setConfigOptions(useOpenGL=True)
    
    # create app
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # define dark color scheme
    dark_bg_color = QColor(40, 40, 40)
    dark_text_color = QColor(200, 200, 200)
    palette = QPalette()
    disabled_color = QColor(100, 100, 100)
    palette.setColor(QPalette.ColorRole.Window, dark_bg_color)
    palette.setColor(QPalette.ColorRole.WindowText, dark_text_color)
    palette.setColor(QPalette.ColorRole.Base, dark_bg_color)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark_bg_color)
    palette.setColor(QPalette.ColorRole.ToolTipBase, dark_bg_color)
    palette.setColor(QPalette.ColorRole.ToolTipText, dark_text_color)
    palette.setColor(QPalette.ColorRole.Text, dark_text_color)
    palette.setColor(QPalette.ColorRole.Button, dark_bg_color)
    palette.setColor(QPalette.ColorRole.ButtonText, dark_text_color)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)

    # create main window
    main = MainWindow()
    main.setPalette(palette)
    main.showMaximized()
    
    app.exec()
