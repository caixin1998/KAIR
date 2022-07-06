import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
from utils import utils_blindsr as blindsr
import json
import torch
def read_json(refer_list_file):
    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)
    return datastore

class XGazeBlindSR(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(XGazeBlindSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize*self.sf


        self.key_to_use = read_json(os.path.join(opt['dataroot_xgaze'], "train_valid_split.json"))["train"]
        self.root = os.path.join(opt['dataroot_xgaze'], "train")
        self.label_path = os.path.join(self.root, "Label")
        self.im_root = os.path.join(self.root, "Image")

        self.path = [os.path.join(self.label_path, path) for path in os.listdir(self.label_path) if path.split('.')[0][-4:] in self.key_to_use]
        self.path.sort()
        self.lines = []

        if isinstance(self.path, list):
            for i in self.path:
                with open(i) as f:
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(self.path) as f:
                self.lines = f.readlines()
                self.lines.pop(0)
        self.selected_lines = []
        for i in range(0,11):
            self.selected_lines += self.lines[i::18]
        self.selected_lines += self.lines[17::18]
        self.lines = self.selected_lines

        # self.paths_H = util.get_image_paths(opt['dataroot_xgaze'])
        print(len(self.lines))

#        for n, v in enumerate(self.paths_H):
#            if 'face' in v:
#                del self.paths_H[n]
#        time.sleep(1)
        assert self.lines, 'Error: H path is empty.'

    def __getitem__(self, index):

        L_path = None

        # ------------------------------------
        # get H image
        # ------------------------------------
        line = self.lines[index]
        line = line.strip().split(" ")
        gaze2d = line[1]
        face_path = line[0]
        # gt_path = face_path
        lmks = np.array(line[3].split(",")).astype("float").reshape(68, 2)
        label = np.array(gaze2d.split(",")).astype("float")
        H_path = os.path.join(self.im_root, face_path)

        img_H = util.imread_uint(H_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        if H < self.patch_size or W < self.patch_size:
            img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8), (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            # rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            # rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            # img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # if 'face' in img_name:
            mode = random.choice([0, 4])
            img_H = util.augment_img(img_H, mode=mode)
            # else:
            #     mode = random.randint(0, 7)
            #     img_H = util.augment_img(img_H, mode=mode)

            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path
        label = torch.from_numpy(label).type(torch.FloatTensor)
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'gaze': label}

    def __len__(self):
        return len(self.lines)
