# Modules to discuss the DU-level trigger and the array-level trigger, and calculate the exposure

This module implements some functions that can be used for the MC simuation data analysis.
These function includes
1. filtering the time traces of GP300 radio antenna detector units (DUs),
2. discussing the FLT0 trigger of each channel (X, Y, & Z channels of each DU), and
3. discussing the array-level trigger (e.g., with criterions that any N DUs or more triggered within a time window of T).
All the functions are written in utils.py.

The following three scripts show how you can use the above functions to get the trigger flag of the simulation events (1 = triggered, 0 = NOT triggered):
1. `judge_trigger_event_level_du_level_channel_level.py`,
2. `judge_trigger_event_level_du_level.py`, and
3. `judge_trigger_event_level.py`.  All the scripts give exactly the same outputs, but they process the time traces of DUs and discuss the FLT0- and array-level triggers in different ways.  
First, `judge_trigger_event_level_du_level_channel_level.py` separately processes each time trace of a DU. It will help you perform flexible processes which may be channel and/or DU dependent.  
Second, `judge_trigger_event_level_du_level.py` processes the time traces of all three channels of a DU together.  
Third, `judge_trigger_event_level.py` processes all DUs toghether. It enables you to write a visually simple code if you want to process all DUs in the same way.  

For the calculation of the exposure, you can use `calculate_exposure.py`. It outputs the figures showing the exposure and the expected number of CR events to be observed in a day, as a function of energy and zenith angle, energy, or zenith angle.
