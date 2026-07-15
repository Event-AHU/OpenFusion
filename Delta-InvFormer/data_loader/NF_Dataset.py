import os
import torch
import random
import h5py
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from .augmentations_video import get_train_joint_transform, get_val_joint_transform
from torchvision import transforms
from PIL import Image
import scipy.io
import time

mean = 41.744637110427604
segma = 47.91132784092548

class NuclearFusion_Dataset(Dataset):
    def __init__(self, mode: str, config: dict) -> None:
        self.config = config
        self.mode = mode
        self.is_training = (mode == "train")
        print("Dataloader Mode:", "training" if self.is_training else "testing")

        # configs
        self.data_root = config["data_root"]
        self.image_folder = config['image_folder']
        self.label_folder = config['label_folder']
        self.image_ext = config['image_ext']
        self.label_ext = config['label_ext']
        # 添加一个反演数据
        self.inversion_folder = config['inversion_folder']
        self.inversion_ext = config['inversion_ext']

        # transform
        if self.is_training:
            self.joint_transform = get_train_joint_transform(scale=config["scale"]) 
            self.time_clips = config['time_clips']
        else:
            self.joint_transform = get_val_joint_transform(scale=config["scale"])
            self.time_clips = config['time_clips']

        # get all frames from video datasets
        self.frame_list, self.path_image, self.path_mask,self.path_inversion = self.generate_images_from_video(is_training=self.is_training)
        ##看看路径
        # print(self.path_image)


        print('Total video clips are {}.'.format(len(self.frame_list)))

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        image_label_path_list = self.frame_list[index]
        
        clip_list = []
        label_list = []
        inversion_list=[]
        w_list = []
        h_list = []
        image_path_list = []
        label_path_list = []
        inversion_path_list = []
        for image_path, label_path, inversion_path in image_label_path_list:
            if not self.is_training:
                image_path = self.path_image[image_path]
                label_path = self.path_mask[label_path]
                inversion_path = self.path_inversion[inversion_path]
                image = Image.open(image_path).convert('RGB')
                label = Image.open(label_path).convert('L')
                inversion = scipy.io.loadmat(inversion_path)
            else:
                image = self.path_image[image_path]
                label = self.path_mask[label_path]
                inversion =self.path_inversion[inversion_path]

            inversion_mat = np.array(inversion['SA_E_result']).clip(min=0,max=600)
            inversion_mat = torch.from_numpy(inversion_mat).to(dtype=torch.float32)

            if inversion_mat.shape == (99,76):
                inversion_mat = inversion_mat.transpose(0,1)
  
            inversion_mat = inversion_mat.unsqueeze(0)  #  (1, H, W)

            inversion_list.append(inversion_mat)

            clip_list.append(image)

            label_list.append(label)
            w, h = image.size
            w_list.append(w)
            h_list.append(h)
            
            image_path_list.append(image_path)
            label_path_list.append(label_path)
            inversion_path_list.append(inversion_path)

        clip_list,label_list= self.joint_transform(clip_list,label_list)
        image_torch=torch.stack(clip_list)
        label_torch=torch.stack(label_list)
        label_torch = label_torch
        inversion_torch=torch.stack(inversion_list)
        return {"image": image_torch, "label": label_torch, "image_path": image_path_list, "inversion":inversion_torch,"label_path": label_path_list, "w": w_list, "h": h_list}

    def generate_images_from_video(self, is_training=True):
        video_list = os.listdir(os.path.join(self.data_root, self.mode, self.image_folder))
        video_frame_dict = {}
        path_frame_dict = {}
        path_mask_dict = {}
        path_inversion_dict = {}

        for video in video_list:
            
            video_path = os.path.join(self.data_root, self.mode, self.image_folder, video)
            frame_list = [os.path.splitext(frame)[0] for frame in os.listdir(video_path) if frame.endswith(self.image_ext)]
            frame_list = self.sort_images(frame_list)

            if self.is_training:
                
                len_frame_list = len(frame_list)
                if len_frame_list < 100:
                    for _ in range(int(100/len_frame_list)+1):
                        for reversed_frame in frame_list[-1:-(min(100-len_frame_list, len_frame_list)):-1]:
                            frame_list.append(reversed_frame)
                    if len(frame_list) >= 100:
                        frame_list = frame_list[:100]

            video_frame_dict[video] = []
            for frame in frame_list:
                
                prefix=""
                frame_path = os.path.join(self.data_root, self.mode, self.image_folder, video, frame + self.image_ext)
                gt_path = os.path.join(self.data_root, self.mode, self.label_folder, video, frame + self.label_ext)
                inversion_path=os.path.join(self.data_root, self.mode, self.inversion_folder,video,prefix+frame+self.inversion_ext)
                frame_gt = (frame_path, gt_path,inversion_path)
                video_frame_dict[video].append(frame_gt)

                if is_training:
                    path_frame_dict[frame_path] = Image.open(frame_path).convert('RGB')
                    path_mask_dict[gt_path] = Image.open(gt_path).convert('L')
                    path_inversion_dict[inversion_path] = scipy.io.loadmat(inversion_path)
                else:
                    path_frame_dict[frame_path] = frame_path
                    path_mask_dict[gt_path] = gt_path
                    path_inversion_dict[inversion_path] = inversion_path
        
        clip_list = []
        for video in video_list:
            frames_from_one_video = video_frame_dict[video]
            stride = 1 if self.is_training else self.time_clips
            for begin in range(0, len(frames_from_one_video) - self.time_clips + 1, stride):
                frame_clips = frames_from_one_video[begin: begin + self.time_clips]
                clip_list.append(frame_clips)

            if self.is_training:
                for begin in range(len(frames_from_one_video) - self.time_clips + 1, len(frames_from_one_video)):
                    frame_clips = frames_from_one_video[begin: begin-self.time_clips: -1]
                    clip_list.append(frame_clips)
            else:
                last_frame_clips = frames_from_one_video[len(frames_from_one_video) - self.time_clips:]
                clip_list.append(last_frame_clips)
        return clip_list, path_frame_dict, path_mask_dict, path_inversion_dict

    def sort_images(self, frame_list):
        frame_int_list = [int(frame) for frame in frame_list]
        # sort images to 001, 002, 003...
        sort_index = [i for i, v in sorted(enumerate(frame_int_list), key=lambda x: x[1])]
        return [frame_list[i] for i in sort_index]

    def sort_inversion(self,inversion_list):
        pass

    def read_segmentation_mask(self, gt_path):
        gt_pil = Image.open(gt_path).convert('L')
        gt_np = np.array(gt_pil)

        # some gt are store in RGB, whose values are not [0, 255]
        if len(np.unique(gt_np)) != 2:
            gt_np[gt_np != 0] = 255

        return Image.fromarray(gt_np)
