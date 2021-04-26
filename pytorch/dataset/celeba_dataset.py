import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import torchvision.transforms as transforms

dataroot = "~/datasets/image/celeba/img_align_celeba"
csv_landmarks = "/home/piotr/Documents/face-mask-detection-pytorch/dataset/list_landmarks_align_celeba.txt"

image_size = 64


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.landmarks_frame = pd.read_csv(csv_file, delim_whitespace=True, engine="python")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.landmarks_frame.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, file_name)
        image = io.imread(img_name)

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        # print(landmarks)
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        # print(landmarks)
        image = self.transform(image)
        plt.imshow(image.permute(1, 2, 0), interpolation='nearest')
        plt.show()
        print("image", image)
        sample = {'image': image, 'landmarks': landmarks}

        # print(image.reshape(-1).shape)
        return image, landmarks
