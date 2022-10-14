import numpy as np
import pc2vox
import time

from os import listdir
from os.path import isfile, join
from scipy import sparse


# Create input data pipeline
pc_names = ['/media/extdrive/rohith/train_data/Arco_Valentino_Dense_vox12.ply', '/media/extdrive/rohith/train_data/Egyptian_mask_vox12.ply', '/media/extdrive/rohith/train_data/Facade_00009_vox12.ply', '/media/extdrive/rohith/train_data/Facade_00015_vox14.ply', '/media/extdrive/rohith/train_data/Facade_00064_vox11.ply', '/media/extdrive/rohith/train_data/Frog_00067_vox12.ply', '/media/extdrive/rohith/train_data/loot_vox10_1200.ply', '/media/extdrive/rohith/train_data/Palazzo_Carignano_Dense_vox14.ply', '/media/extdrive/rohith/train_data/redandblack_vox10_1550.ply', '/media/extdrive/rohith/train_data/Shiva_00035_vox20.ply', '/media/extdrive/rohith/train_data/soldier_vox10_0690.ply', '/media/extdrive/rohith/train_data/ULB_Unicorn_vox13_n.ply']
#pc_names = ['/media/extdrive/rohith/train_data/Head_00039_vox12.ply']

total_blocks=[]
for i in range(len(pc_names)):
    print(pc_names[i])
    # Load input PC, get list of coordinates
    in_points = pc2vox.load_pc(pc_names[i])
    # Divide PC into blocks of the desired size. Get list of relative coordinates for points in each block
    blocks, _ = pc2vox.pc2blocks(in_points, 64)
    # Ignore blocks with fewer than 500 points
    total_blocks.extend([blk for blk in blocks if len(blk) >= 500])
    print(len(blocks))

vox_data = np.zeros([len(total_blocks), 64, 64, 64, 1], dtype=np.float32)
# Iterate all blocks
for j in range(len(total_blocks)):
    # Convert coordinates to 3D block
    vox_data[j, :, :, :, :] = pc2vox.point2vox(total_blocks[j], 64)
    
print("start file creation")
# Compression
for k in range(len(vox_data)):
    np.save("/media/extdrive/rohith/train_data_np/second_half_block_%d"%(k), vox_data[k])
    
