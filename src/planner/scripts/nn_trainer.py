'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-05-03 22:31:45
'''
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

VECTOR_SIZE = 24
OUTPUT_SIZE = 9
IMG_WIDTH = 160
IMG_HEIGHT = 120


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
                transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
                transforms.ToTensor(),
            ])
            img_tensor = transform(depth_img)

            # read csv data and convert it to tensor
            vector = torch.tensor(row[1:-9].values.astype(np.float32))  # 1:-9 means from 1 to -9 (not included)
            output = torch.tensor(row[-9:].values.astype(np.float32))  # len: 9

            # concatenate the two tensors and form the input and output
            inputs.append((img_tensor, vector))  # inputs[i][0] is the image, inputs[i][1] is the vector
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


class PlannerNet(nn.Module):
    def __init__(self):
        super(PlannerNet, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, IMG_WIDTH, IMG_WIDTH)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (32, IMG_WIDTH, IMG_WIDTH)
            nn.ReLU(),  # activation
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(  # input shape (32, IMG_WIDTH, IMG_WIDTH)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # output shape (64, IMG_WIDTH/2, IMG_WIDTH/2)
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(  # input shape (64, IMG_WIDTH/2, IMG_WIDTH/2)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # output shape (128, IMG_WIDTH/4, IMG_WIDTH/4)
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.img_mlp = nn.Sequential(
            nn.Linear(128 * IMG_WIDTH * IMG_HEIGHT // 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, VECTOR_SIZE),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(VECTOR_SIZE + VECTOR_SIZE, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_SIZE)
        )

    def forward(self, input):
        img = input[0]
        vector = input[1]

        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 128 * IMG_WIDTH * IMG_HEIGHT // 16)
        x = self.img_mlp(x)

        x = torch.cat([x, vector], dim=1)
        x = self.mlp(x)

        return x


class Trainer():
    def __init__(self):
        pass

    def build_dataset(self):
        # read data from local files
        data_reader = DataReader()
        inputs, outputs = data_reader.load_data()

        # generate the dataset
        plan_dataset = PlanDataset(inputs, outputs)
        print("shape of plan_dataset: ", len(plan_dataset))

        # split the dataset into training set and validation set
        train_size = int(0.8 * len(plan_dataset))
        val_size = len(plan_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(plan_dataset, [train_size, val_size])
        print("shape of train_dataset: ", len(train_dataset))
        print("shape of val_dataset: ", len(val_dataset))

        # generate the dataloader
        self.train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        self.val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

    def train(self):
        self.planner_net = PlannerNet()
        self.criterion = nn.MSELoss()
        optimizer = optim.Adam(self.planner_net.parameters(), lr=0.001)
        num_epochs = 3

        # train the network
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                inputs, expert_outputs = data
                optimizer.zero_grad()

                # forward
                outputs = self.planner_net(inputs)
                loss = self.criterion(outputs, expert_outputs)

                # backward
                loss.backward()
                optimizer.step()

                # training loss
                running_loss += loss.item()

            # print the training loss of each epoch
            print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(self.train_dataloader)))

        # save the trained model
        torch.save(self.planner_net.state_dict(), 'saved_net/planner_net.pth')

    def test(self):
        planner_net_test = PlannerNet()
        planner_net_test.load_state_dict(torch.load('saved_net/planner_net.pth'))
        total_loss = 0.0
        for i, data in enumerate(self.val_dataloader, 0):
            inputs, expert_outputs = data
            outputs = planner_net_test(inputs)
            loss = self.criterion(outputs, expert_outputs)
            total_loss += loss.item()

        print('Validation loss: %.3f' % (total_loss / len(self.val_dataloader)))


if __name__ == '__main__':

    trainer = Trainer()
    trainer.build_dataset()
    trainer.train()
    trainer.test()

    # generate a random input
    # input = torch.randn(1, 1, IMG_WIDTH, IMG_HEIGHT)
    # vector = torch.randn(1, VECTOR_SIZE)
    # time_start = time.time()
    # output = self.planner_net((input, vector))
    # time_end = time.time()
    # print("time cost: ", time_end - time_start)
    # print("output shape: ", output.shape)

    # load the trained model
    #
