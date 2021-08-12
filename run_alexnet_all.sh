#!/usr/bin/env bash
args_list=(
#"3e4 20 True"
#"3.5e4 20 True"
#"4e4 20 True"
#"4.5e4 20 True"
#"5e4 20 True"
#
#
#"3e4 20 False"
#"3.5e4 20 False"
#"4e4 20 False"
#"4.5e4 20 False"
#"5e4 20 False"
#

"1e4 20 False"
"1.5e4 20 False"
"2e4 20 False"
"2.5e4 20 False"


"1e4 20 True"
"1.5e4 20 True"
"2e4 20 True"
"2.5e4 20 True"



)
#type_list=['norm1','norm2','sample_L1','sample_L2','none']
for args_string in $"${args_list[@]}"
do
args=(${args_string})
export PB=${args[0]}
echo ${PB}
export TYPE='sample_L2'
echo ${TYPE}
export CLIP_BOUND=${args[1]}
echo ${CLIP_BOUND}
export USE_NEW_METHOD=${args[2]}
echo "USE_NEW_METHOD"
echo ${USE_NEW_METHOD}
export RATIO_TYPE=0
echo "RATIO_TYPE"
echo ${RATIO_TYPE}

if [ "$USE_NEW_METHOD"x = "True"x ];
then
   export LOG_PATH="log/Alexnet/"${TYPE}"/"${USE_NEW_METHOD}"/"${RATIO_TYPE}
   echo "true"
else
   export LOG_PATH="log/Alexnet/"${TYPE}"/"${USE_NEW_METHOD}
   echo "False"
fi



echo ${LOG_PATH}
bash run_alexnet.sh
done

