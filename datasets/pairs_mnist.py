import os
import numpy as np
import cv2
import torch
import torchvision
import math
import colorsys
from skimage.transform import resize
from copy import deepcopy
from utils.misc import get_RTmat
from utils.misc import freq_to_wave
import pdb
import copy



class Pair_MNIST(SequentialMNIST):
    def __init__(
            self,
            root,
            train=True,
            transforms=torchvision.transforms.ToTensor(),
            T=3,
            max_angle_velocity_ratio=[-0.2, 0.2],
            max_angle_accl_ratio=[-0.0, 0.0],
            max_color_velocity_ratio=[-0.2, 0.2],
            max_color_accl_ratio=[-0.0, 0.0],
            max_pos=[-8, 8],
            max_trans_accl=[-10.0, 10.0],
            label=False,
            label_velo=False,
            label_accl=False,
            max_T=9,
            only_use_digit4=False,
            backgrnd=False,
            shared_transition=False,
            pair_transition=False,
            same_object=True,
            rng=None,
            fixpos=True):
        super().__init__(root,
            train=train,
            transforms=transforms,
            T=T,
            max_angle_velocity_ratio=max_angle_velocity_ratio,
            max_angle_accl_ratio=max_angle_accl_ratio,
            max_color_velocity_ratio=max_color_velocity_ratio,
            max_color_accl_ratio=max_color_accl_ratio,
            max_pos=max_pos,
            max_trans_accl=max_trans_accl,
            label=label,
            label_velo=label_velo,
            label_accl=label_accl,
            max_T=max_T,
            only_use_digit4=only_use_digit4,
            backgrnd=backgrnd,
            shared_transition=shared_transition,
            rng=rng,
            )

        self.first_indices = np.random.choice(len(self.data), len(self.data))
        self.second_indices = np.random.choice(len(self.data), len(self.data))
        self.pair_transition = pair_transition
        self.same_object = same_object
        self.fixpos = fixpos

        if self.fixpos == True:
            np.random.seed(5)
            self.pairpos0 , self.pairpos1 = self.initial_poss()

    def initial_poss(self):

        pos0 = [] #pos0_obj0, pos0_obj1
        pos1 = [] #pos1_obj0, pos1_obj1

        pos0.append(self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2]))
        pos0.append(self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2]))

        pos1.append(self.rng.uniform(pos0[0], self.max_pos[0]-pos0[0], size=[2]))
        pos1.append(self.rng.uniform(pos0[1], self.max_pos[1]-pos0[1], size=[2]))

        return pos0, pos1

    def __getitem__(self, i):

        if self.fixpos == True:
            pos0, pos1 = self.pairpos0, self.pairpos1
        else:
            pos0, pos1 = self.initial_poss()


        if self.same_object == True:
            first_idx = 1
            second_idx = 4
        else:
            first_idx = self.first_indices[i]
            second_idx = self.second_indices[i]


        # If running experiment with all sequences consisting of the same object
        params0 = self.otain_action_parameters(pos0[0], pos1[0])
        params1 = copy.deepcopy(params0)

        #param1 is the same as param0 all except at initial position, angle and color
        params1['pos0'], params1['pos1'] = pos0[1], pos1[1]
        angle_0 = self.rng.uniform(0, 2 * math.pi, size=1)
        color_0 = self.rng.uniform(0, 1, size=1)
        params1['angles_0'], params1['color_0'] = angle_0, color_0

        images0 = self.get_image(first_idx, **params0)
        images1 = self.get_image(second_idx, **params1)

        images0_seq = []
        images1_seq = []

        for t in range(len(images0)):
            _image0 = images0[t]
            image0 = self.add_backgrnd(_image0, first_idx)
            _image1 = images1[t]
            image1 = self.add_backgrnd(_image1, second_idx)

            images0_seq.append(image0)
            images1_seq.append(image1)


        return (images0_seq, images1_seq)

    def add_backgrnd(self, _image,  i):
        if hasattr(self, 'backgrnd_data'):
            _imagemask = (np.sum(_image, axis=2, keepdims=True) < 3e-1)
            _image = torch.tensor(
                _image) + self.backgrnd_data[i].permute([1, 2, 0]) * (
                         _imagemask)
        image = np.array(torch.clip(_image, max=1.))
        return image


    def otain_action_parameters(self, pos1, pos0):
        angles_0 = self.rng.uniform(0, 2 * math.pi, size=1)
        color_0 = self.rng.uniform(0, 1, size=1)

        if self.shared_transition:
            (angles_v, angles_a) = (self.angles_v, self.angles_a)
            (color_v, color_a) = (self.color_v, self.color_a)
            (pos_v, pos_a) = (self.pos_v, self.pos_a)
        else:
            angles_v = self.rng.uniform(
                math.pi * self.angle_velocity_range[0],
                math.pi * self.angle_velocity_range[1], size=1)
            angles_a = self.rng.uniform(math.pi * self.angle_accl_range[0],
                                        math.pi * self.angle_accl_range[1],
                                        size=1)
            color_v = 0.5 * self.rng.uniform(self.color_velocity_range[0],
                                             self.color_velocity_range[1],
                                             size=1)
            color_a = 0.5 * \
                      self.rng.uniform(
                          self.color_accl_range[0],
                          self.color_accl_range[1], size=1)
            pos_v = (pos1 - pos0) / (self.max_T - 1)
            pos_a = self.rng.uniform(
                self.max_trans_accl[0], self.max_trans_accl[1], size=[2])

        action_params={}
        action_params['angles_0'] = angles_0
        action_params['angles_v'] = angles_v
        action_params['angles_a'] = angles_a
        action_params['color_0'] = color_0
        action_params['color_v'] = color_v
        action_params['color_a'] = color_a
        action_params['pos0'] = pos0
        action_params['pos_v'] = pos_v
        action_params['pos_a'] = pos_a
        action_params['pos1'] = pos1

        return action_params



    def get_image(self, i, pos0,
                  pos1,
                  angles_0,
                  angles_v,
                  angles_a,
                  color_0,
                  color_v,
                  color_a,
                  pos_v,
                  pos_a,
                  immobile=False):
        original_size = 28
        digit_size = 14
        margin = int((32 - digit_size)/2)

        image = np.array(self.data[i][0], np.float32).reshape(original_size, original_size)
        image = resize(image, [digit_size , digit_size])
        image = cv2.copyMakeBorder(
            image, margin , margin , margin , margin , cv2.BORDER_CONSTANT, value=(0, 0, 0))

        images = []

        for t in range(self.T):
            if immobile == False:
                angles_t = 0.5 * angles_a * t ** 2 + angles_v * t + angles_0
                ###
                # Freeze translation
                ###
                #pos_t = 0.5 * pos_a * t ** 2 + pos_v * t + pos0
                pos_t = pos0
                color_t = (0.5 * color_a * t ** 2 + t * color_v + color_0) % 1
            if immobile == True:
                angles_t = angles_0
                pos_t = pos0
                color_t = color_0

            mat = get_RTmat(0, 0, float(angles_t), 32, 32, pos_t[0],
                            pos_t[1])
            _image = cv2.warpPerspective(image.copy(), mat, (32, 32))

            rgb = np.asarray(colorsys.hsv_to_rgb(
                color_t, 1, 1), dtype=np.float32)
            _image = np.concatenate(
                [_image[:, :, None]] * 3, axis=-1) * rgb[None, None]
            _image = _image / 255.

            images.append(self.transforms(_image.astype(np.float32)))

        if self.label or self.label_velo:
            ret = [images]
            if self.label:
                ret += [self.data[i][1]]
            if self.label_velo:
                ret += [
                    freq_to_wave(angles_v.astype(np.float32)),
                    freq_to_wave(
                        (2 * math.pi * color_v).astype(np.float32)),
                    pos_v.astype(np.float32)
                ]
            if self.label_accl:
                ret += [
                    freq_to_wave(angles_a.astype(np.float32)),
                    freq_to_wave(
                        (2 * math.pi * color_a).astype(np.float32)),
                    pos_a.astype(np.float32)
                ]
            return ret
        else:
            return images

