import numpy as np
from open3d import *    

def main():
    data = "/media/extdrive/rohith/test_data/longdress_vox10_1300.ply"
    cloud = read_point_cloud(data) # Read the point cloud
    draw_geometries([cloud]) # Visualize the point cloud     

if __name__ == "__main__":
    main()