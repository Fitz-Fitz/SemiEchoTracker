import os
import torch
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np
from torchvision import transforms as T
import random
import albumentations as A

class EchoDataset(Dataset):
    def __init__(self, root_dir="", transform=None, frame_num=10, splits='train',
                 target_size=(224, 224), data_type="PLAX",
                 rotation_limit=30.0, rotation_prob=0.5):
        target_data_dir = os.path.join(root_dir, data_type)
        if not os.path.exists(target_data_dir):
            raise FileNotFoundError(f"Directory {target_data_dir} does not exist")
        self.root_dir = target_data_dir
        self.transform = transform
        self.normalize = T.Compose([
            T.ToTensor(), 
            T.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225])
        ])
        self.target_size = target_size
        self.frame_num = frame_num
        self.splits = splits
        self.data_type = data_type
        self.rotation_limit = float(rotation_limit)
        self.rotation_prob = float(rotation_prob)
        self.data_list = self.read_data()
        self.rotation = splits == 'train' and self.rotation_limit > 0 and self.rotation_prob > 0
        if len(self.data_list) == 0:
            raise ValueError(f"No valid data found in {self.root_dir}")
        print(f'{len(self.data_list)} patients selected for {self.splits} set in {self.root_dir}')
        
    def read_data(self):
        print(f'Reading data from {self.root_dir}/{self.splits}.txt')
        with open(f'{self.root_dir}/{self.splits}.txt', 'r') as f:
            patient_list = f.read().splitlines()
        data_all = []
        print(len(patient_list))
        for patient in patient_list:
            patient_dir = os.path.join(self.root_dir, "selected_data", patient)
            if not os.path.isdir(patient_dir):
                continue
            anno_file_list = glob.glob(f'{self.root_dir}/selected_data/{patient}/data.npz')
            base_data_anno = np.load(anno_file_list[0], allow_pickle=True)
            landmarks = base_data_anno['landmarks']
            image_list = base_data_anno['video']
            data_all.append({
                'patient': patient,
                'landmarks': landmarks,
                'image_list': image_list
            })
        return data_all

    def __len__(self):
        return len(self.data_list)

    def rotate_landmarks(self, landmarks, angle, center):
        """
        Rotate landmarks around center point
        Args:
            landmarks: array of shape (N, 2) containing x,y coordinates
            angle: rotation angle in degrees
            center: tuple of (x,y) rotation center coordinates
        """
        angle_rad = np.deg2rad(angle)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        landmarks = landmarks - center
        landmarks = landmarks @ rot_matrix
        landmarks = landmarks + center
        return landmarks

    def __getitem__(self, idx):
        meta_data = self.data_list[idx]
        patient_name = meta_data['patient']
        image_list = meta_data['image_list']
        landmarks_raw = np.array(meta_data['landmarks'])
        has_framewise_landmarks = len(landmarks_raw) == len(image_list)
        has_endpoint_landmarks = len(landmarks_raw) == 2

        # Decide the clip-level rotation angle (applied to every frame / landmark)
        if self.rotation:
            self.current_angle = random.uniform(-self.rotation_limit, self.rotation_limit) \
                if random.random() < self.rotation_prob else 0

        # Whether the transform supports replay (ReplayCompose). We apply the
        # same pixel-level augmentation to every frame so the appearance is
        # consistent across time (critical for the tracker).
        replay_params = None
        is_replay_compose = isinstance(self.transform, A.ReplayCompose)

        input_data = []
        scale_x = scale_y = rescale_x = rescale_y = None
        for index, image in enumerate(image_list):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                orig_h, orig_w = image.shape[:2]
                scale_x = self.target_size[1] / orig_w
                scale_y = self.target_size[0] / orig_h
                rescale_x = orig_w / self.target_size[1]
                rescale_y = orig_h / self.target_size[0]
                image = cv2.resize(image, self.target_size)

                if self.rotation:
                    image = A.rotate(image, angle=self.current_angle)

                if is_replay_compose:
                    if replay_params is None:
                        aug = self.transform(image=image)
                        replay_params = aug['replay']
                    else:
                        aug = A.ReplayCompose.replay(replay_params, image=image)
                    image = aug['image']
                else:
                    aug = self.transform(image=image)
                    image = aug['image']
            image = self.normalize(image)
            input_data.append(image)
        input_data = torch.stack(input_data)
        landmarks = landmarks_raw * np.array([scale_x, scale_y])
        
        # Apply rotation to landmarks if needed
        if self.rotation:
            center = np.array([self.target_size[1]/2, self.target_size[0]/2])  # rotation center (x,y)
            # Rotate both ED and ES frames
            landmarks[0] = self.rotate_landmarks(landmarks[0], self.current_angle, center)
            landmarks[-1] = self.rotate_landmarks(landmarks[-1], self.current_angle, center)
            
        # check if need reverse
        if self.data_type == "A4C":
            area_first_frame = cv2.contourArea(landmarks[0].astype(np.int32))
            area_last_frame = cv2.contourArea(landmarks[-1].astype(np.int32))
            if area_first_frame > area_last_frame:
                landmarks = np.flip(landmarks, axis=0).copy()
                input_data = torch.flip(input_data, dims=[0]).contiguous()
        elif self.data_type == "PLAX":
            check_index_1, check_index_2 = 3, 6
            dis_first_frame = np.sqrt((landmarks_raw[0, check_index_1,0] - landmarks_raw[0, check_index_2,0])**2 + (landmarks_raw[0, check_index_1,1] - landmarks_raw[0, check_index_2,1])**2)
            dis_last_frame  = np.sqrt((landmarks_raw[-1, check_index_1,0] - landmarks_raw[-1, check_index_2,0])**2 + (landmarks_raw[-1, check_index_1,1] - landmarks_raw[-1, check_index_2,1])**2)
            if dis_first_frame > dis_last_frame:
                landmarks = np.flip(landmarks, axis=0).copy()
                input_data = torch.flip(input_data, dims=[0]).contiguous()
        # sample self.frame_num
        if len(input_data) > self.frame_num:
            if self.frame_num >= 2:
                middle_indices = np.linspace(1, len(input_data)-2, self.frame_num-2, dtype=int)
                indices = np.concatenate([[0], middle_indices, [len(input_data)-1]])
            else:
                indices = np.linspace(0, len(input_data)-1, self.frame_num, dtype=int)
            input_data = input_data[indices]
            if has_framewise_landmarks:
                landmarks = landmarks[indices]
            elif has_endpoint_landmarks:
                landmarks = landmarks[[0, -1]]
            else:
                raise ValueError(
                    f"Unsupported landmark count {len(landmarks)} for {len(image_list)} frames in {patient_name}"
                )
        if self.splits == 'train':
            landmarks = landmarks[[0, -1]]
        return {
            'video': input_data,
            'landmarks': landmarks,
            'patient_name': patient_name,
            'rescale_x': rescale_x,
            'rescale_y': rescale_y
        }
