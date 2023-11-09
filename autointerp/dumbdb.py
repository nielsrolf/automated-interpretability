import glob
import os
import pickle


class ExperimentData:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.items = self._load()

    def _load(self):
        # Find all .pkl files in the save directory
        files = glob.glob(os.path.join(self.save_dir, "*.pkl"))
        # Extract version numbers from the filenames
        versions = [int(os.path.basename(f).split(".")[0]) for f in files]
        # If there are no files, return an empty dict
        if not versions:
            return []
        # Find the latest version
        latest_version = max(versions)
        # Load the latest version
        with open(os.path.join(self.save_dir, f"{latest_version:05d}.pkl"), "rb") as f:
            return pickle.load(f)

    def __call__(self, item):
        self.items += [item]
        return item

    def filter(self, **filters):
        res = []
        for item in self.items:
            if self._check_filter(item, filters):
                res += [item]
        return res

    def _check_filter(self, v, condition):
        # print('checking', v, condition, type(condition))
        try:
            callable = condition(v)
        except:
            callable = False
        if v == condition or callable:
            return True
        if isinstance(condition, dict) and isinstance(v, dict):
            for k_, c_ in condition.items():
                if not self._check_filter(v.get(k_), c_):
                    # c is a dict of conditions that is False because c_ is False
                    return False
                # the entire dict of conditions is checked and all are true
            return True
        else:
            return False

    def to_disk(self):
        # Find the next version number
        files = glob.glob(os.path.join(self.save_dir, "*.pkl"))
        versions = [int(os.path.basename(f).split(".")[0]) for f in files]
        next_version = max(versions) + 1 if versions else 0
        # Save the new version
        with open(os.path.join(self.save_dir, f"{next_version:05d}.pkl"), "wb") as f:
            pickle.dump(self.items, f)


store = ExperimentData("/Users/nielswarncke/Documents/code/TransformerLens/experiments/new/")
