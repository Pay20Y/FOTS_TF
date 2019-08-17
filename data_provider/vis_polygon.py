import cv2
import os
import numpy as np
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, help="images dir")
parser.add_argument("-g", "--gt_dir", type=str, help="gt dir")
parser.add_argument('-i', '--is_17', type=bool, default=False)
parser.add_argument('-s', '--save_dir', type=str)
args = parser.parse_args()

images_root_dir = args.data_dir
gt_root_dir = args.gt_dir

for path in tqdm.tqdm(os.listdir(images_root_dir)):
    img = cv2.imread(os.path.join(images_root_dir, path))
    gt_path = 'gt_' + path.replace('jpg', 'txt')
    with open(os.path.join(gt_root_dir, gt_path), "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(',')
            coord = np.array(list(map(eval, line[:8]))).reshape((-1, 2))
            cv2.polylines(img, [coord.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
    cv2.imwrite(os.path.join(args.save_dir, path), img)

