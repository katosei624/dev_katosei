import sys
import grand.dataio.data_handling as groot
import grand.manage_log as mlg
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
from utils import *



FLT0_trig_params_file = './dict_trig_params_fir.csv'
#FLT0_trig_params_file = sys.argv[1]
#directory = '/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_0000'
directory = '/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_0010'
#directory = sys.argv[2]
d_input = groot.DataDirectory(directory)
output  = open(directory[directory.find('sim'):],'w')

trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0

event_list = tadc_l1.get_list_of_events()
nb_events  = len(event_list)

if nb_events == 0:
    sys.exit("No events in the file. Exiting.")

####################################################################################
# start looping over the events
####################################################################################
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
    tadc_trace = np.array(tadc_l1.trace_ch) # dimension: (# of du_id, XYZ, samples)

    f_sample = 500e6 # Hz, ADC sampling rate
    t_res = int(1. / f_sample * 1.e9) # ns, ADC time resolution
    rel_trace_start_time = calculate_relative_trace_start_time(tshower_l0, tadc_l1, t_res)

    trig_du_list = []
  
    for du_id_n in range(du_id.shape[0]): # loop for DUs

        tadc_trace_X = tadc_trace[du_id_n][0]
        tadc_trace_Y = tadc_trace[du_id_n][1]
        tadc_trace_Z = tadc_trace[du_id_n][2]
        rel_start_time = rel_trace_start_time[du_id_n]

        ### Filtering of the traces
        ### The following lines applies
        ### 1. notch filter with a notch frequency of 39 MHz, &
        ### 2. FIR filter only passing the signals below 115 MHz
        tadc_X_filt = notch_filter(tadc_trace_X, 39e6, 0.9, f_sample)
        tadc_Y_filt = notch_filter(tadc_trace_Y, 39e6, 0.9, f_sample)
        tadc_Z_filt = notch_filter(tadc_trace_Z, 39e6, 0.9, f_sample)

        tadc_X_filt = filter_traces_bandpass(tadc_X_filt, coeff_file='./lowpass115MHz.txt')
        tadc_Y_filt = filter_traces_bandpass(tadc_Y_filt, coeff_file='./lowpass115MHz.txt')
        tadc_Z_filt = filter_traces_bandpass(tadc_Z_filt, coeff_file='./lowpass115MHz.txt')
        #if events <= 10 :
        #    for n in range(tadc_X_filt.shape[0]):
        #        print(tadc_X_filt[n], tadc_Y_filt[n], tadc_Z_filt[n])

        ### Discuss the FLT0 trigger in the channel level
        ### The function "trigger_FLT0" is used (developed by M. Guelfand),
        ### but here we use a wrap function which gives a relative trigger time(s) of the DU of interest.
        FLT0_trig_params = get_FLT0_trigger_parameters(FLT0_trig_params_file)
        FLT0_trig_time_X = get_FLT0_trigger_time(tadc_X_filt, FLT0_trig_params, rel_start_time, t_res)
        FLT0_trig_time_Y = get_FLT0_trigger_time(tadc_Y_filt, FLT0_trig_params, rel_start_time, t_res)
        FLT0_trig_time_Z = get_FLT0_trigger_time(tadc_Z_filt, FLT0_trig_params, rel_start_time, t_res)
        #for n in FLT0_trig_time_X:
        #    print(n)
        #for n in FLT0_trig_time_Y:
        #    print(n)
        #for n in FLT0_trig_time_Z:
        #    print(n)

        ### Make a list of the relative trigger time(s) of all channels
        for trig_time_m in FLT0_trig_time_X: trig_du_list.append([du_id[du_id_n], trig_time_m, 'X'])
        for trig_time_m in FLT0_trig_time_Y: trig_du_list.append([du_id[du_id_n], trig_time_m, 'Y'])
        #for trig_time_m in FLT0_trig_time_Z: trig_du_list.append([du_id[du_id_n], trig_time_m, 'Z'])

        ### END of loop for DUs

    #if events >= 50: break
    # Discuss a coincidence between DUs
    any_du = 4 # or more DUs
    t_window = 1e4 # ns
    arry_trig, arry_trig_du_list = discuss_coincidence(trig_du_list, any_du, t_window)

    output.write('{:d}'.format(tshower_l0.run_number)+' ')
    output.write('{:9d}'.format(tshower_l0.event_number)+' ')
    output.write('{:9s}'.format(tshower_l0.primary_type)+' ')
    output.write('{:9.2e}'.format(tshower_l0.energy_primary)+' ')
    output.write('{:9.2f}'.format(tshower_l0.zenith)+' ')
    output.write('{:9.2f}'.format(tshower_l0.azimuth)+' ')
    output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[0])+' ')
    output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[1])+' ')
    output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[2])+' ')
    #output.write('{:6d}'.format(arry_trig)+'\n')
    output.write('{:6d}'.format(arry_trig)+' ')
    output.write(str(arry_trig_du_list)+'\n')
    events += 1
    #if events >= 2: break
    #if events >= 5: break
    #if events >= 10: break
    #if events >= 20: break

print('End of judge_trigger_with_ADCtrace_like_experiment.py')
