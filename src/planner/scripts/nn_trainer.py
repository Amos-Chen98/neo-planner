'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-06-06 17:30:20
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
from torchinfo import summary
import time
import torch.onnx


VECTOR_SIZE = 24
OUTPUT_SIZE = 9
IMG_WIDTH = 160
IMG_HEIGHT = 120
BATCH_SIZE = 32
EPOCHS = 20


class DataReader():
    def __init__(self, img_path='training_data/depth_img', csv_path='training_data/train.csv'):
        self.img_path = img_path
        self.csv_path = csv_path

    def load_data(self):  # sourcery skip: avoid-builtin-shadow
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
            img_flatten = img_tensor.reshape(-1)  # flatten the image tensor

            # read csv data and convert it to tensor
            vector = torch.tensor(row[1:-9].values.astype(np.float32))  # 1:-9 means from 1 to -9 (not included)
            input = torch.cat((img_flatten, vector))  # concatenate the two tensors and form the input
            output = torch.tensor(row[-9:].values.astype(np.float32))  # len: 9

            inputs.append(input)
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
        self.conv1 = nn.Sequential(  # input shape (1, IMG_WIDTH, IMG_HEIGHT)
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
        self.conv2 = nn.Sequential(  # input shape (32, IMG_WIDTH, IMG_HEIGHT)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # output shape (64, IMG_WIDTH/2, IMG_HEIGHT/2)
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(  # input shape (64, IMG_WIDTH/2, IMG_HEIGHT/2)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # output shape (128, IMG_WIDTH/4, IMG_HEIGHT/4)
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
        # retrieve the image and vector from the input
        img = input[:, :IMG_WIDTH * IMG_HEIGHT].reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT)
        vector = input[:, IMG_WIDTH * IMG_HEIGHT:]

        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 128 * IMG_WIDTH * IMG_HEIGHT // 16)
        x = self.img_mlp(x)

        x = torch.cat([x, vector], dim=1)
        x = self.mlp(x)

        return x


class NetOperator():
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

    def build_dataset(self):
        # read data from local files
        data_reader = DataReader()
        inputs, outputs = data_reader.load_data()

        # generate the dataset
        plan_dataset = PlanDataset(inputs, outputs)
        print("Len of whole dataset: ", len(plan_dataset))

        # split the dataset into training set and test set
        train_size = int(0.8 * len(plan_dataset))
        test_size = len(plan_dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(plan_dataset, [train_size, test_size])
        print("Len of train dataset: ", len(train_set))
        print("Len of test dataset: ", len(test_set))

        # generate the dataloader
        self.train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)
        self.test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)
        print("DataLoader generated.")

    def init_net(self):
        self.planner_net = PlannerNet()
        self.planner_net.to(self.device)  # move the network to GPU
        # print summary of the network
        summary(self.planner_net, (BATCH_SIZE, IMG_WIDTH * IMG_HEIGHT + VECTOR_SIZE), device=self.device)

    def train_and_save_net(self):
        optimizer = optim.Adam(self.planner_net.parameters(), lr=0.001)

        # train the self.planner_net
        print("Start training...")

        # train the network
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                # move data to GPU
                data = [d.to(self.device) for d in data]

                inputs, expert_outputs = data  # TODO

                optimizer.zero_grad()

                # forward
                outputs = self.planner_net(inputs)
                loss = self.criterion(outputs, expert_outputs)
                # print("inputs: ", inputs)
                # print("outputs: ", outputs)
                # print("expert_outputs: ", expert_outputs)

                # backward
                loss.backward()
                optimizer.step()

                # training loss
                running_loss += loss.item()

            # print the training loss of each epoch
            print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(self.train_dataloader)))

        # save the trained model
        torch.save(self.planner_net.state_dict(), 'saved_net/planner_net.pth')  # save a pth model
        self.save_onnx()  # save an onnx model

    def save_onnx(self):
        dummy_input = torch.randn(1, IMG_WIDTH*IMG_HEIGHT+VECTOR_SIZE)
        dummy_input = dummy_input.to(self.device)  # move the dummy input to GPU
        torch.onnx.export(self.planner_net,
                          dummy_input,
                          "saved_net/planner_net.onnx",
                          input_names=['input'],
                          output_names=['output']
                          )
        print("planner_net.onnx saved!")

    def load_and_test_net(self):
        planner_net_test = PlannerNet()
        planner_net_test.load_state_dict(torch.load('saved_net/planner_net.pth'))
        total_loss = 0.0
        for i, data in enumerate(self.test_dataloader, 0):
            inputs, expert_outputs = data
            outputs = planner_net_test(inputs)
            loss = self.criterion(outputs, expert_outputs)
            total_loss += loss.item()

        print('Test loss: %.3f' % (total_loss / len(self.test_dataloader)))


if __name__ == '__main__':

    net_operator = NetOperator()
    net_operator.build_dataset()
    net_operator.init_net()

    time_start = time.time()
    net_operator.train_and_save_net()
    time_end = time.time()
    training_time = time_end - time_start

    # convert the training time to hour:minute:second
    m, s = divmod(training_time, 60)
    h, m = divmod(m, 60)
    print("\nTraining time cost: %d:%02d:%02d\n" % (h, m, s))

    net_operator.load_and_test_net()

    # generate a random input
    # planner_net = PlannerNet()
    # img = torch.randn(1, 1, IMG_WIDTH, IMG_HEIGHT)
    # vector = torch.randn(1, VECTOR_SIZE)
    # time_start = time.time()
    # output = planner_net((img, vector))
    # time_end = time.time()
    # print("time cost: ", time_end - time_start)
    # print("output shape: ", output.shape)
