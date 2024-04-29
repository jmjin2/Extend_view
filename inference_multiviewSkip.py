import argparse
import cv2
import glob
import os
import shutil
import torch
from arch.multiviewSkip_arch import MultiViewSkipSR
from utils.util import tensor2img, read_img_seq

def inference(imgs_list, imgnames, model, save_path):
    with torch.no_grad():
        outputs = model(imgs_list[0], imgs_list[1], imgs_list[2])
    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_Multiview_conv1x1.png'), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/MultiviewSkipSR_ch/net_g_30000.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/MIV_val/MIVx4/(W01)Group/textureAllViews', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='./results/skip_models/Group/30000_ch', help='save image path')
    parser.add_argument('--interval', type=int, default=15, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = MultiViewSkipSR(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # extract images from video format files
    # multiview frame 추가

    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

    # load data and inference
    view_list = os.listdir(args.input_path)
    view1 = sorted(glob.glob(os.path.join(input_path, view_list[0], '*')))
    view2 = sorted(glob.glob(os.path.join(input_path, view_list[1], '*')))
    view3 = sorted(glob.glob(os.path.join(input_path, view_list[2], '*')))
    mv_imgs_list = [view1, view2, view3]

    # for i in view_list:
    #     mv_imgs_list.append(sorted(glob.glob(os.path.join(input_path, i, '*'))))

    num_frame = len(view1)
    if num_frame <= args.interval:  # too many images may cause CUDA out of memory
        imgs_list = []
        for i in range(len(view_list)):
            if i==0:
                lq1, imgnames = read_img_seq(mv_imgs_list[i], return_imgname=True)
                lq1 = lq1.unsqueeze(0).to(device)
            if i==1:
                lq2, imgnames = read_img_seq(mv_imgs_list[i], return_imgname=True)
                lq2 = lq2.unsqueeze(0).to(device)
            if i==2:
                lq3, imgnames = read_img_seq(mv_imgs_list[i], return_imgname=True)
                lq3 = lq3.unsqueeze(0).to(device)

            imgs_list.append(lq1)
            imgs_list.append(lq2)
            imgs_list.append(lq3)

        inference(imgs_list, imgnames, model, args.save_path)

    else:
        for idx in range(0, num_frame, args.interval):
            imgs_list = []
            interval = min(args.interval, num_frame - idx)

            for i in range(len(view_list)):
                if i==0:
                    lq1, imgnames = read_img_seq(mv_imgs_list[i][idx:idx + interval], return_imgname=True)
                    lq1 = lq1.unsqueeze(0).to(device)
                if i==1:
                    lq2, imgnames = read_img_seq(mv_imgs_list[i][idx:idx + interval], return_imgname=True)
                    lq2 = lq2.unsqueeze(0).to(device)
                if i==2:
                    lq3, imgnames = read_img_seq(mv_imgs_list[i][idx:idx + interval], return_imgname=True)
                    lq3 = lq3.unsqueeze(0).to(device)
            imgs_list.append(lq1)
            imgs_list.append(lq2)
            imgs_list.append(lq3)

            inference(imgs_list, imgnames, model, args.save_path)

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    main()
