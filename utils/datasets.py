from pathlib import Path

import h5py
import torch


class RobosuiteDataset(torch.utils.data.Dataset):
    def __init__(self, task):
        demos = self.load_robosuite_demo(task)
        self.obs = []
        self.actions = []
        for demo in demos:
            for i in range(len(demo["obs"])):
                self.obs.append(demo["obs"][i])
                self.actions.append(demo["action"][i])

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
        }

    @staticmethod
    def load_robosuite_demo(task):
        demo_path = Path(f"datasets/robosuite/{task}.hdf5")
        data = []
        with h5py.File(demo_path, "r") as f:
            data_group = f["data"]
            for demo_id in data_group.keys():
                demo = data_group[demo_id]
                data.append(RobosuiteDataset._process_robosuite_demo(demo))
        return data

    @staticmethod
    def _process_robosuite_demo(demo):
        obs = torch.from_numpy(demo["states"][()]).float()
        actions = torch.from_numpy(demo["actions"][()]).float()
        rewards = torch.from_numpy(demo["rewards"][()]).float()
        dones = torch.from_numpy(demo["dones"][()]).float()
        return {"obs": obs, "action": actions, "reward": rewards, "terminated": dones}
