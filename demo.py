import sys
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
import time
from model import *
from utils import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument and global variables
parser = argparse.ArgumentParser('Interface')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--num_fc', type=int, default=1, choices=[1, 2], help='# of fully-connected layer')
parser.add_argument('--fc_dim', type=int, default=32, help='fully-connected layer dimension')

parser.add_argument('--patch_size', type=int, default=32, help='patch size')
parser.add_argument('--ol_ratio', type=float, default=0.5, help='overlapping ratio')
parser.add_argument('--max_round', type=int, default=10, help='max round for early stopper')
parser.add_argument('--cal_method', type=str, default='all', choices=['max', 'mean', 'any', 'all'], help='calculating method in probability')
parser.add_argument('--th', type=float, default=0.5, help='threshold in calculating mask')

try:
    args = parser.parse_args()

except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_FC = 1
FC_DIM = None
PATCH_SIZE = args.patch_size
OL_RATIO = args.ol_ratio

NUM_EPOCH = args.n_epoch
MAX_ROUND = args.max_round
CAL_METHOD = args.cal_method
TH = args.th

test_list = pd.read_csv('test_list.csv', index_col=0)

MODEL_NUM = 0

spec = test_list.loc[MODEL_NUM]
BATCH_SIZE, LEARNING_RATE, NUM_FC, FC_DIM, PATCH_SIZE, OL_RATIO = spec[:]
BATCH_SIZE = int(BATCH_SIZE)
LEARNING_RATE = float(LEARNING_RATE)
NUM_FC = int(NUM_FC)
FC_DIM = int(FC_DIM)
PATCH_SIZE = int(PATCH_SIZE)
OL_RATIO = float(OL_RATIO)

PRETRAINED_MODEL = str(MODEL_NUM).zfill(3)
PRETRAINED_PATH = f'./saved_models/{PRETRAINED_MODEL}.pth'

model = Model(num_fc=NUM_FC,
              fc_dim=FC_DIM
              )

def load_demo_dataset(data_type):
    paths = []
    raw_path = f'./dataset/{data_type}/img/'
    img_ = lambda x: f'./dataset/{data_type}/img/{x[:-4]}.jpg'

    for file_num in os.listdir(raw_path):
        img_path = img_(file_num)
        paths.append(img_path)

    return np.array(paths)

def demo(model, images, visualizing):
    for img_p in images:
        print(f'Processing - {img_p}')
        narrow_img, patches, _, patch_indices_map, _ = \
            transform_img(img_p, [], PATCH_SIZE, OL_RATIO)

        discriminator = Discriminate(narrow_img.shape[1:], CAL_METHOD, TH)
        patch_indices = np.arange(patches.shape[0])
        num_instance = patches.shape[0]
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        for k in tqdm(range(num_batch)):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance, s_idx + BATCH_SIZE)

            patch_cut = patch_indices[s_idx:e_idx]
            target_patches = patches[patch_cut]
            mask_cut = patch_indices_map[patch_cut]

            pred_prob = model(target_patches).cpu().detach().numpy()
            discriminator.fill(mask_cut, pred_prob)

        output = discriminator.score().cpu()

        if visualizing:
            visualize(img_p, narrow_img, _, output, 1)


if __name__ == '__main__':
    print(f'Loading the best model | MODEL NUM : {PRETRAINED_MODEL}')
    model.load_state_dict(torch.load(PRETRAINED_PATH))
    model.to(device)
    model.eval()
    print(f'Loaded')
    print(f'Probability Calculating Method : {CAL_METHOD}')
    test_images = load_demo_dataset('demo')
    demo(model, test_images, 1)
    print(f'Done')