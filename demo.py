import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# from scipy.misc import imresize
# from skimage.transform import resize
from cv2 import resize
from model import ModelSpatial
from utils import imutils, evaluation
from config import *
import cv2

parser = argparse.ArgumentParser()
# parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
parser.add_argument('--model_weights', type=str, help='model weights', default='trained_model/epoch_18_weights.pt')
parser.add_argument('--image_dir', type=str, help='images', default='data/gaze_follow_test/frames')
parser.add_argument('--head', type=str, help='head bounding boxes', default='data/gaze_follow_test/test.txt')
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=200)
args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def run():
    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(args.head, names=column_names, index_col=0)
    df['left'] -= (df['right']-df['left'])*0.1
    df['right'] += (df['right']-df['left'])*0.1
    df['top'] -= (df['bottom']-df['top'])*0.1
    df['bottom'] += (df['bottom']-df['top'])*0.1

    # set up data transformation
    test_transforms = _get_transform()

    model = ModelSpatial()
    model_dict = model.state_dict()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    pretrained_dict = torch.load(args.model_weights,
                                 map_location=device)

    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.train(False)

    with torch.no_grad():
        for i in df.index:
            frame_raw = Image.open(os.path.join(args.image_dir, i))
            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size

            head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]

            head = frame_raw.crop((head_box)) # head crop

            head = test_transforms(head) # transform inputs
            frame = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=input_resolution).unsqueeze(0)

            head = head.unsqueeze(0)
            frame = frame.unsqueeze(0)
            head_channel = head_channel.unsqueeze(0)

            # forward pass
            raw_hm, _, inout = model(frame, head_channel, head)
            # heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255
            norm_map = resize(raw_hm, (height, width)) - inout

            print(inout)
            # # MyVis
            head_box = list(map(int, head_box))
            frame_raw = np.array(frame_raw)
            cv2.rectangle(frame_raw, (head_box[0], head_box[1]), (head_box[2], head_box[3]), (255, 0, 0), 2)
            if inout < args.out_threshold:  # in-frame gaze
                pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                norm_p = [pred_x / output_resolution, pred_y / output_resolution]
                cv2.circle(frame_raw, (int(norm_p[0] * width), int(norm_p[1] * height)), int(height / 50.0),  (255,0,0), -1)
                # cv2.line(frame_raw, (int(norm_p[0] * width), int((head_box[0] + head_box[2]) / 2)),
                #          (int(norm_p[1] * height), int((head_box[1] + head_box[3]) / 2)), (0,255,0), 2)
                starting_point = ((head_box[0] + head_box[2]) // 2, (head_box[1] + head_box[3]) // 2)
                ending_point = (int(norm_p[0] * width), int(norm_p[1] * height))
                cv2.line(frame_raw, starting_point, ending_point,(255, 0, 0), 5)

            screen = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            cv2.imshow('Frame', screen)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        print('DONE!')


if __name__ == "__main__":
    run()
