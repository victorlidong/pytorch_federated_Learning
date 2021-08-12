#!/usr/bin/env bash
python main_alexnet.py --type=${TYPE} --pb=${PB} --clip_bound=${CLIP_BOUND} --batch_size=64 --test_batch_size=64 --log_interval=1 --epochs=1  --num_rounds=10 --is_ratio_list=${USE_NEW_METHOD} --ratio_type=${RATIO_TYPE} --log_path=${LOG_PATH}
