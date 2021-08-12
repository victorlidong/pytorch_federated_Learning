#!/usr/bin/env bash

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

#"1e4 20 True 0"
#"1e3 20 True 0"
#"1e2 20 True 0"
#"5e3 20 True 0"
#"5e2 20 True 0"




"50 20 True 0"
"50 20 True 1"
"50 20 True 2"
"50 20 True 3"
"50 20 True 4"
"50 20 True 5"
"50 20 True 6"
"50 20 True 7"
"50 20 True 8"
"50 20 True 9"
"50 20 True 10"
"50 20 True 11"
"50 20 True 12"
"50 20 True 13"
"50 20 True 14"
"50 20 True 15"

)
#type_list=['norm1','norm2','sample_L1','sample_L2','none']
for args_string in $"${args_list[@]}"
do
args=(${args_string})
export PB=${args[0]}
echo ${PB}
export TYPE='norm2'
echo ${TYPE}
export CLIP_BOUND=${args[1]}
echo ${CLIP_BOUND}
export USE_NEW_METHOD=${args[2]}
echo "USE_NEW_METHOD"
echo ${USE_NEW_METHOD}
export RATIO_TYPE=0
echo "RATIO_TYPE"
echo ${RATIO_TYPE}

export TEST_LAYER=${args[3]}
echo "TEST_LAYER"
echo ${TEST_LAYER}


if [ "$USE_NEW_METHOD"x = "True"x ];
then
   export LOG_PATH="log/test/Alexnet/"${TYPE}"/"${USE_NEW_METHOD}"/"${RATIO_TYPE}
   echo "True"
else
   export LOG_PATH="log/test/Alexnet/"${TYPE}"/"${USE_NEW_METHOD}
   echo "False"
fi



echo ${LOG_PATH}

python main_alexnet.py --type=${TYPE} --pb=${PB} --clip_bound=${CLIP_BOUND} --batch_size=64 --test_batch_size=64 --log_interval=1 --epochs=1  --num_rounds=10 --is_ratio_list=${USE_NEW_METHOD} --ratio_type=${RATIO_TYPE} --log_path=${LOG_PATH} --test_layer=${TEST_LAYER}

done



