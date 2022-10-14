#!/bin/bash


echo $(python ADLPCC.py train "/media/extdrive/rohith/train_data_np" "/media/extdrive/rohith/models_dynamic_data_batch4/900/70" --lambda 900 --fl_alpha 0.7 --batchsize 4 --last_step 1000000)


echo $(python ADLPCC.py train "/media/extdrive/rohith/train_data_np" "/media/extdrive/rohith/models_dynamic_data_batch4/1500/70" --lambda 1500 --fl_alpha 0.7 --batchsize 4 --last_step 1000000)


echo $(python ADLPCC.py train "/media/extdrive/rohith/train_data_np" "/media/extdrive/rohith/models_dynamic_data_batch4/5000/70" --lambda 5000 --fl_alpha 0.7 --batchsize 4 --last_step 1000000)


