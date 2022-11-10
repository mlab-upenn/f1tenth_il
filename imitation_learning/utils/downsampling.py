

from concurrent.futures import process


def downsample(data, observation_shape, downsampling_method):
    
    if downsampling_method == "simple":
        # print("observation_shape type: ", type(observation_shape))
        # print("observation_shape: ", observation_shape)
        obs_gap = int(1080/observation_shape)
        processed_data = data[::obs_gap]
    else:
        processed_data = data
    return processed_data
