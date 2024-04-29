import os
import os.path as osp
from PIL import Image
from util import scandir

def generate_meta_info_MIV():

    gt_folder = 'datasets/MIV/GT'
    meta_info_txt = 'data/meta_info_MIV_GT.txt'
    idx = len(gt_folder) +1

    with open(meta_info_txt, 'w') as f:
        # for file in os.listdir(gt_folder):
        #     print(file)

        for path, _, files in os.walk(gt_folder):
            if "depth" in path: 
                continue
            img_list = sorted(list(scandir(path)))
            frame_num=0
            width=0
            height=0
            n_channel=0
            for _, img_path in enumerate(img_list):
                if img_path.endswith('.jpg'):
                    img = Image.open(osp.join(path, img_path))  # lazy load
                    width, height = img.size
                    mode = img.mode
                    if mode == 'RGB':
                        n_channel = 3
                    elif mode == 'L':
                        n_channel = 1
                    else:
                        raise ValueError(f'Unsupported mode {mode}.')
                    frame_num+=1
            if frame_num != 0:
                info = f'{path[idx:]} {frame_num} ({height},{width},{n_channel})'
                print(frame_num, info)
                f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info_MIV()
