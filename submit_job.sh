#!/bin/sh

thisdir=/pbs/home/s/skato/grand/work/sim/Lukas/RFChain_v2/dev_katosei
exe=${thisdir}/exe
input=${thisdir}/COREAS-AN_sim_data_directory

mkdir -p ${exe}

for n in `cat ${input}`
#for n in `cat ${input} | head -2 | tail -1`
#for n in `cat ${input} | head -1`
#for n in `cat ${input} | head -10`
do
    num=`echo $n | cut -c 103-106`
    cat ${thisdir}/job.sh | sed 's/filenumber/'$num'/g' > ${exe}/exe-$num.sh
    sbatch ${exe}/exe-$num.sh
done
