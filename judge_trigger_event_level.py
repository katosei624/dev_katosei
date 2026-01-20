import sys
import grand.dataio.data_handling as groot
import grand.manage_log as mlg
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
from utils import *

### How to execute the script:
### $ python judge_trigger_event_level.py data_directory FLT0_trig_params_file
### Example:
### $ python judge_trigger_event_level.py /your/data/directory ./dict_trig_params_fir.csv

### Set input data and an output file
### data_directory: directory where you have GRAND root files
### FLT0_trig_params_file: file containing the FLT0 parameters,
### please see dict_trig_params_fir.csv for its format
data_directory, FLT0_trig_params_file = sys.argv[1], sys.argv[2]

### The output will be generated in the directory './out_judge_trigger',
### so please make sure to create the directory before executing the script.
output  = open('out_judge_trigger/'+data_directory[data_directory.find('sim'):],'w')

### Read GRAND root data
d_input = groot.DataDirectory(data_directory)
trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0

event_list = tadc_l1.get_list_of_events()
nb_events  = len(event_list)
if nb_events == 0: sys.exit("No events in the file. Exiting.")

### Start event loop
previous_run = None
events = 0

for event_number,run_number in event_list: # loop for events

    assert isinstance(event_number, int)
    assert isinstance(run_number, int)

    tadc_l1.get_event(event_number, run_number)
    tshower_l0.get_event(event_number, run_number)
    
    if previous_run != run_number:
        trun_l1.get_run(run_number)
        previous_run = run_number
        
    du_id = np.array(tadc_l1.du_id)
    tadc_trace = np.array(tadc_l1.trace_ch) # dimension: (N_du, XYZ, samples)
    
    f_sample = 500e6 # Hz, ADC sampling rate
    t_res = int(1. / f_sample * 1.e9) # ns, ADC time resolution
    rel_trace_start_time = calculate_relative_trace_start_time(tshower_l0, tadc_l1, t_res)

    ### Filtering of the traces
    ### The following lines applies
    ### 1. notch filter with a notch frequency of 39 MHz, &
    ### 2. FIR filter only passing the signals below 115 MHz
    trace_filt = notch_filter_all_dus(tadc_trace, 39e6, 0.9, f_sample)
    trace_filt = filter_traces_bandpass_all_dus(trace_filt, coeff_file='./lowpass115MHz.txt')

    ### Discuss the FLT0 trigger in the channel level
    ### The function "trigger_FLT0" is used (developed by M. Guelfand),
    ### but here we use a wrap function which gives a relative trigger time(s) of the DU of interest.
    FLT0_trig_params = get_FLT0_trigger_parameters(FLT0_trig_params_file)
    trig_chnl_list   = get_FLT0_trigger_time_all_dus(du_id, trace_filt, FLT0_trig_params, rel_trace_start_time, t_res)
        
    ### Discuss a coincidence between DUs
    ### A condition to claim a coincidence is
    ### any any_du DUs or more within a time window of t_window nanoseconds
    any_du = 4
    t_window = 1e4
    used_channels = ['X', 'Y'] # Now only X & Y channels are used to discuss a coincidence
    arry_trig, arry_trig_chnl_list = discuss_coincidence(trig_chnl_list, any_du, t_window, used_channels)

    output.write('{:d}'.format(tshower_l0.run_number)+' ')
    output.write('{:9d}'.format(tshower_l0.event_number)+' ')
    output.write('{:9s}'.format(tshower_l0.primary_type)+' ')
    output.write('{:9.2e}'.format(tshower_l0.energy_primary)+' ')
    output.write('{:9.2f}'.format(tshower_l0.zenith)+' ')
    output.write('{:9.2f}'.format(tshower_l0.azimuth)+' ')
    output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[0])+' ')
    output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[1])+' ')
    output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[2])+' ')
    output.write('{:6d}'.format(arry_trig)+'\n')
    #output.write('{:6d}'.format(arry_trig)+' ')
    #output.write(str(arry_trig_chnl_list)+'\n')

print('judge_trigger_event_level.py ended')
