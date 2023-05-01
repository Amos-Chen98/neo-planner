import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class DataReader():
    def __init__(self, img_path='training_data/depth_img', csv_path='training_data/train.csv'):
        self.img_path = img_path
        self.csv_path = csv_path

    def load_data(self):
        csv_data = pd.read_csv(self.csv_path)
        inputs = []
        outputs = []
        for index, row in csv_data.iterrows():
            timestamp = int(row['id'])
            # read depth image and convert it to tensor
            img_file_name = os.path.join(self.img_path, f'{timestamp}.png')
            depth_img = Image.open(img_file_name)
            transform = transforms.Compose([
                transforms.Resize((640, 480)),
                transforms.ToTensor(),
            ])
            img_tensor = transform(depth_img)

            # read csv data and convert it to tensor
            vector = torch.tensor(row[1:-9].values.astype(np.float32))  # 1:-9 means from 1 to -9 (not included)
            output = torch.tensor(row[-9:].values.astype(np.float32))  # len: 9

            # concatenate the two tensors and form the input and output
            inputs.append((img_tensor, vector))
            outputs.append(output)

        return inputs, outputs


class PlanDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# class TrajNet(torch.nn.Module):
#     def __init__(self):
#         super(DroneNet, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = torch.nn.Linear(64 * 16 * 16 + 6, 256)
#         self.fc2 = torch.nn.Linear(256, 128)
#         self.fc3 = torch.nn.Linear(128, 6)

#     def forward(self, depth, input_data):
#         x = self.conv1(depth)
#         x = torch.nn.functional.relu(x)
#         x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = self.conv2(x)
#         x = torch.nn.functional.relu(x)
#         x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = x.view(-1, 64 * 16 * 16)
#         x = torch.cat([x, input_data], dim=1)
#         x = self.fc1(x)
#         x = torch.nn.functional.relu(x)
#         x = self.fc2(x)
#         x = torch.nn.functional.relu(x)
#         x = self.fc3(x)
#         return x


if __name__ == '__main__':
    data_reader = DataReader()
    inputs, outputs = data_reader.load_data()
    plan_dataset = PlanDataset(inputs, outputs)
    print("shape of plan_dataset: ", len(plan_dataset))

    # split the dataset into training set and validation set
    train_size = int(0.8 * len(plan_dataset))
    val_size = len(plan_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(plan_dataset, [train_size, val_size])
    print("shape of train_dataset: ", len(train_dataset))
    print("shape of val_dataset: ", len(val_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
