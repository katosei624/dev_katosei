#!/bin/bash

# SLURM options:

#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --output=log/%j.log   # Standard output and error log

#SBATCH --partition=htc               # Partition choice (htc by default)

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=2000                    # Memory in MB per default
#SBATCH --time=1-00:00:00             # Max time limit = 7 days
#SBATCH --licenses=sps                # Declaration of storage and/or software resources

data_dir=/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_filenumber
trig_params=dict_trig_params_fir.csv
thisdir=/pbs/home/s/skato/grand/work/sim/Lukas/RFChain_v2/dev_katosei
log=${thisdir}/log
#mkdir -p ${log}

python ${thisdir}/judge_trigger_event_level.py ${data_dir} ${trig_params}
