#!/bin/bash

<<com
#printf "Basketball player"

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/basketball_player_vox11_00000200.ply" "/media/extdrive/rohith/models_dynamic_data/500/70" --blk_size 64

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/basketball_player_vox11_00000200.ply" "/media/extdrive/rohith/models_half_data/900/*" --blk_size 64

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/basketball_player_vox11_00000200.ply" "/media/extdrive/rohith/models_half_data/500/*" --blk_size 64

#printf "Statue"

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/Staue_Klimt_vox12.ply" "/media/extdrive/rohith/models_half_data/1500/*" --blk_size 64

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/Staue_Klimt_vox12.ply" "/media/extdrive/rohith/models_half_data/900/*" --blk_size 64

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/Staue_Klimt_vox12.ply" "/media/extdrive/rohith/models_half_data/500/*" --blk_size 64

#printf "Queen"

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_half_data/1500/*" --blk_size 64

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_half_data/900/*" --blk_size 64

#python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_half_data/500/*" --blk_size 64


printf "House without roof"

python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_half_data/1500/*" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_half_data/900/*" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_half_data/500/*" --blk_size 64

printf "Dancer"

python ADLPCC.py compress "/media/extdrive/rohith/test_data/dancer_vox11.ply" "/media/extdrive/rohith/models_half_data/1500/*" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/dancer_vox11.ply" "/media/extdrive/rohith/models_half_data/900/*" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/dancer_vox11.ply" "/media/extdrive/rohith/models_half_data/500/*" --blk_size 64


printf "Long dress"

python ADLPCC.py compress "/media/extdrive/rohith/test_data/longdress_vox10_1300_n.ply" "/media/extdrive/rohith/models_half_data/1500/*" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/longdress_vox10_1300_n.ply" "/media/extdrive/rohith/models_half_data/900/*" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/longdress_vox10_1300_n.ply" "/media/extdrive/rohith/models_half_data/500/*" --blk_size 64


python ADLPCC.py compress "/media/extdrive/rohith/test_data/longdress_vox10_1300_n.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64
python ADLPCC.py compress "/media/extdrive/rohith/test_data/dancer_vox11.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64
python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64
python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64
python ADLPCC.py compress "/media/extdrive/rohith/test_data/Staue_Klimt_vox12.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64
python ADLPCC.py compress "/media/extdrive/rohith/test_data/basketball_player_vox11_00000200.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/Staue_Klimt_vox12_downsample.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64
python ADLPCC.py compress "/media/extdrive/rohith/test_data/Staue_Klimt_vox12_upsample.ply" "/media/extdrive/rohith/models_half_data/2000/*" --blk_size 64


python ADLPCC.py compress "/media/extdrive/rohith/test_data/basketball_player_vox11_00000200.ply" "/media/extdrive/rohith/models_dynamic_data/1500/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/basketball_player_vox11_00000200.ply" "/media/extdrive/rohith/models_dynamic_data/5000/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/basketball_player_vox11_00000200.ply" "/media/extdrive/rohith/models_dynamic_data/20000/70" --blk_size 64
com

#printf "Queen"

python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/500/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/900/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/1500/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/queen_frame_0200_n.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/5000/70" --blk_size 64

#printf "House without roof"

python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/500/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/900/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/1500/70" --blk_size 64

python ADLPCC.py compress "/media/extdrive/rohith/test_data/House_without_roof_00057_vox12.ply" "/media/extdrive/rohith/models_dynamic_data_batch8/5000/70" --blk_size 64
