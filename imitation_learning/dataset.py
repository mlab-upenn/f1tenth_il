import numpy as np
import utils.downsampling as downsampling

class Dataset(object):
    def __init__(self):
        self.observs = None
        self.poses_x = None
        self.poses_y = None
        self.poses_theta = None
        self.scans = None 
        self.actions = None

    def add(self, data):
        assert data["observs"].shape[0] == data["poses_x"].shape[0]
        assert data["observs"].shape[0] == data["poses_y"].shape[0]
        assert data["observs"].shape[0] == data["poses_theta"].shape[0]
        assert data["observs"].shape[0] == data["scans"].shape[0]
        assert data["observs"].shape[0] == data["actions"].shape[0]
        
        if self.observs is None:
            self.observs = data["observs"]
            self.poses_x = data["poses_x"]
            self.poses_y = data["poses_y"]
            self.poses_theta = data["poses_theta"]
            self.scans = data["scans"]
            self.actions = data["actions"]
        else:
            self.observs = np.concatenate([self.observs, data["observs"]])
            self.poses_x = np.concatenate([self.poses_x, data["poses_x"]])
            self.poses_y = np.concatenate([self.poses_y, data["poses_y"]])
            self.poses_theta = np.concatenate([self.poses_theta, data["poses_theta"]])
            self.scans = np.concatenate([self.scans, data["scans"]])
            self.actions = np.concatenate([self.actions, data["actions"]])

    def sample(self, batch_size):
        idx = np.random.permutation(self.scans.shape[0])[:batch_size]
        return {"scans":self.scans[idx], "actions":self.actions[idx]}
    
    def get_num_of_total_samples(self):
        return self.scans.shape[0]