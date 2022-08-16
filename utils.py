import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

def load_dataset(dir):
    paths = []

    pristine = [os.path.join(dir,'0' ,f) for f in os.listdir(os.path.join(dir,'0')) if os.path.isfile(os.path.join(dir,'0', f))]
    forged = [os.path.join(dir,'1' ,f) for f in os.listdir(os.path.join(dir,'1')) if os.path.isfile(os.path.join(dir,'1', f))]

    paths.extend(pristine)
    paths.extend(forged)
    labels = np.concatenate((np.zeros(len(pristine)), (np.ones(len(forged)))))

    return paths, labels

def transform_img(path):
    img = Image.open(path)
    img = F.to_tensor(img)
    return torch.unsqueeze(img, dim=0).to(device)

class EarlyStopMonitor(object):
    def __init__(self, max_round, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = -1
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.is_best = False

    def early_stop_check(self, curr_val):
        self.epoch_count += 1

        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.last_best = curr_val
            self.best_epoch = self.epoch_count
            self.is_best = True

        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            self.is_best = True

        else:
            self.num_round += 1
            self.is_best = False

        return self.num_round >= self.max_round

