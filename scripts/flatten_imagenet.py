import os 
import os.path as osp

import shutil

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from functools import partial

from PIL import Image


def check_image_size(file):
    img = Image.open(file)
    h, w = img.size

    if h < 256 or w < 256:
        return None
    else:
        return file
    

def copy_file(file, dir_to):
    shutil.copy(file, osp.join(dir_to, osp.basename(file)))


def main(dir_from, dir_to):
    # check duplicated
    files = []
    signate_dirs = ['train', 'val']
    sub_dirs = []
    for signate_dir in signate_dirs:
        sub_dirs.extend(
            [osp.join(signate_dir, sub_dir) for sub_dir in os.listdir(osp.join(dir_from, signate_dir)) if osp.isdir(osp.join(dir_from, signate_dir, sub_dir))]
        )
    for sub_dir in tqdm(sub_dirs):
        files.extend([osp.join(dir_from, sub_dir, f) for f in os.listdir(osp.join(dir_from, sub_dir))])

    # check image size
    cur_len = len(files)
    files = process_map(check_image_size, files, max_workers=64, chunksize=1)
    files = [f for f in files if f is not None]
    print(f'{cur_len - len(files)} files are removed')

    # copy files
    f = partial(copy_file, dir_to=dir_to)
    process_map(f, files, max_workers=64, chunksize=1)
    


if __name__ == '__main__':
    dir_from = '/data/datasets/imagenet'
    dir_to = '/data/datasets/imagenet_flatten'

    if not osp.exists(dir_to):
        os.makedirs(dir_to)

    main(dir_from, dir_to)
