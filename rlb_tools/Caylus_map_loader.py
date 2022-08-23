
import cv2

import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def load_maps(
    hard_obstacles: bool= True,
    dense_vegetation: bool = True, 
    light_vegetation: bool = True, 
    paths: bool = True, 
    scale: float=1.0) -> list:
    
    def load_feature_map_array(path: str):
        # -> Load image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # -> Set transparent values as white
        alpha_channel = img[:, :, 3]
        _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
        color = img[:, :, :3]
        new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))

        # -> Convert to binary image
        gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        ret, bw_img = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY)

        return bw_img

    # -> Initialise grids dict
    grid_dict = {}

    # -> load all requested grids
    from rlb_config.simulation_parameters import hard_obstacles_image_path, dense_vegetation_img_path, light_vegetation_img_path, paths_image_path

    if hard_obstacles:
        grid_dict["hard_obstacles"] = 1-load_feature_map_array(path=hard_obstacles_image_path)

    if dense_vegetation:
        grid_dict["dense_vegetation"] = 1-load_feature_map_array(path=dense_vegetation_img_path)

    if light_vegetation:
        grid_dict["light_vegetation"] = 1-load_feature_map_array(path=light_vegetation_img_path)

    if paths:
        grid_dict["paths"] = load_feature_map_array(path=paths_image_path)
    
    for grid in grid_dict.values():
        grid = cv2.resize(grid, (0,0), fx = scale, fy = scale)

    return grid_dict

# load_maps(scale=0.1)