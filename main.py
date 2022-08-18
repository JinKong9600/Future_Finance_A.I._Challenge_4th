from model import *
from utils import *
import math
import sys
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument and global variables
parser = argparse.ArgumentParser('Interface')
parser.add_argument('--train_data', type=int, default=40, help='train data path')
parser.add_argument('--patch_size', type=int, default=40, help='patch size')
parser.add_argument('--ol_ratio', type=float, default=0.5, help='overlapping ratio')

parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--num_fc', type=int, default=1, choices=[1, 2], help='# of fully-connected layer')
parser.add_argument('--fc_dim', type=int, default=32, help='fully-connected layer dimension')
parser.add_argument('--max_round', type=int, default=10, help='max round for early stopper')

try:
    args = parser.parse_args()

except:
    parser.print_help()
    sys.exit(0)

PATCH_SIZE = args.patch_size
OL_RATIO = args.ol_ratio
NUM_EPOCH = args.n_epoch
BATCH_SIZE = args.batch_size
LR = args.lr
NUM_FC = 1
FC_DIM = None
MAX_ROUND = args.max_round

model = Model(num_fc=NUM_FC,
              fc_dim=FC_DIM
              ).to(device)
criterion = nn.CrossEntropyLoss
early_stopper = EarlyStopMonitor(MAX_ROUND)


def eval_one_epoch(mode):
    images, forgery_box, labels = load_dataset(mode)

def train():
    images, forgery_box, labels = load_dataset('train')

    for epoch in range(NUM_EPOCH):
        images, forgery_box, labels = random_shuffle(images, forgery_box, labels)

        for img_p, f_box in zip(images, forgery_box):
            print(img_p)
            patches = transform_img(img_p, f_box, PATCH_SIZE, OL_RATIO)
            print(f'RAW PATCHES : {patches.shape}')
            patch_indices = np.arange(len(patches))
            num_instance = patches.shape[0]
            num_batch = math.ceil(num_instance / BATCH_SIZE)
            print(f'Num Patches : {num_instance}')

            np.random.shuffle(patch_indices)
            for k in range(num_batch):
                s_idx = k * BATCH_SIZE
                e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)

                patch_cut = patch_indices[s_idx:e_idx]
                target_patches = patches[patch_cut]
                # target_labels = patch_cut

                output = model(target_patches)

if __name__ == '__main__':
    train()
