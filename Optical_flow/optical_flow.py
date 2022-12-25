import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import csv

index = 0
DEVICE = 'cuda'
u_list = []
v_list = []


def load_image(imfile,u,v,exp):
    img = Image.open(imfile)
    if exp == 1:
        x= 1900
        y=1300
        w=700
        h=400
    else:
        x= 1100
        y=700
        w=700
        h=400 

    img = np.array(img).astype(np.uint8)
    crop_img = img[y+int(v):y+h+int(v),x+int(u):x+w+int(u)]
    img = crop_img

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    global index
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    u,v,flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    img_flo = img_flo[:, :, [2,1,0]]
    return u,v


def main(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('./models/raft.pth'))

    model = model.module
    model.to(DEVICE)
    model.eval()
    u = 0
    v = 0
    u_list.append(u)
    v_list.append(v)
    image_num = 0
    if args.exp == 1:
        path = './frames/exp1/'
    else:
        path = './frames/exp2/'
    with torch.no_grad():
        images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.JPG'))
        
        #print(images)
        images = sorted(images)
        image_num = len(images)

        for index,(imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            
            image1 = load_image(imfile1,u,v,args.exp)
            image2 = load_image(imfile2,u,v,args.exp)
            

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=30, test_mode=True)
            u,v = viz(image1, flow_up)
            u_list.append(u)
            v_list.append(v)
    if args.exp == 1:
        scale_factor = 0.055
        name = 'exp1_'
    else:
        scale_factor = 0.1
        name = 'exp2_'
    with open(name+'relative.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['u(cm)','v(cm)'])
        for i in range(image_num):
            writer.writerow([u_list[i]*scale_factor,v_list[i]*scale_factor])

    with open(name+'absolute.csv','w') as f:
        accumulate_u  = 0
        accumulate_v = 0
        writer = csv.writer(f)
        writer.writerow(['u(cm)','v(cm)'])
        for i in range(image_num):
            accumulate_u += u_list[i]*scale_factor
            accumulate_v += v_list[i]*scale_factor
            writer.writerow([accumulate_u,accumulate_v])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int,required=True,help="dataset for evaluation")
    args = parser.parse_args()
    if args.exp != 1 and args.exp !=2:
        print('exp argument must be 1 or 2')
    else:
        main(args)

