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

def transform_img(path, patch_size, overlapping_ratio):
    img = Image.open(path)
    img = F.to_tensor(img)

    (h, w) = img.shape[1:]
    lcm = np.lcm(8, patch_size)
    h_cut, w_cut = (h//lcm)*lcm, (w//lcm)*lcm
    upper_h_cut, left_w_cut = round((h-h_cut)/2), round((w-w_cut)/2)
    img = torch.narrow(img, 1, upper_h_cut, h_cut)
    img = torch.narrow(img, 2, left_w_cut, w_cut)
    img = img.unsqueeze(0)

    patches = img.unfold(2, patch_size, int(patch_size*overlapping_ratio))\
        .unfold(3, patch_size, int(patch_size*overlapping_ratio))

    overlapping_patch = torch.empty((patches.shape[2]*patches.shape[3], 3, patch_size, patch_size))
    c = 0
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            overlapping_patch[c] = patches[:, :, i, j, :]
            c += 1

    return overlapping_patch

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

