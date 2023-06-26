'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-06-26 11:42:32
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
import torchvision
import torchvision.models as models


IMG_WIDTH = 160
IMG_HEIGHT = 120
VECTOR_SIZE = 24
OUTPUT_SIZE = 9
BATCH_SIZE = 64
EPOCHS = 3

current_path = os.path.dirname(os.path.abspath(__file__))[:-8]  # -7 remove '/scripts'
img_path = '/training_data/depth_img'
csv_path = '/training_data/train.csv'
pth_save_path = '/saved_net/planner_net.pth'
onnx_save_path = '/saved_net/planner_net.onnx'


class DataReader():
    def __init__(self):

        self.img_path = current_path + img_path
        self.csv_path = current_path + csv_path

        print("img_path: ", self.img_path)
        print("csv_path: ", self.csv_path)

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

        self.resnet = models.resnet18(weights='DEFAULT')

        for param in self.resnet.parameters():
            # freeze the parameters
            param.requires_grad = False

        # change the input channel to 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change the output size to fit VECTOR_SIZE, self.resnet.fc.in_features means the layer's original input size
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, VECTOR_SIZE)

        self.mlp = nn.Sequential(
            nn.Linear(VECTOR_SIZE + VECTOR_SIZE, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUT_SIZE)
        )

    def forward(self, input):
        # retrieve the image and vector from the input
        img = input[:, :IMG_WIDTH * IMG_HEIGHT].reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT)
        vector = input[:, IMG_WIDTH * IMG_HEIGHT:]

        img_feature = self.resnet(img)
        x = torch.cat([img_feature, vector], dim=1)
        x = self.mlp(x)

        return x


class NetOperator():
    def __init__(self):
        print(f"PyTorch version: {torch.__version__}")
        print(f"torchvision version: {torchvision.__version__}")
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.pth_save_path = current_path + pth_save_path
        self.onnx_save_path = current_path + onnx_save_path

        print("pth_save_path: ", self.pth_save_path)
        print("onnx_save_path: ", self.onnx_save_path)

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

        print("Start training...")

        # train the network
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                data = [d.to(self.device) for d in data]  # move data to GPU
                inputs, expert_outputs = data

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
        torch.save(self.planner_net.state_dict(), self.pth_save_path)  # save a pth model
        self.save_onnx()  # save an onnx model

    def save_onnx(self):
        dummy_input = torch.randn(1, IMG_WIDTH*IMG_HEIGHT+VECTOR_SIZE)
        dummy_input = dummy_input.to(self.device)  # move the dummy input to GPU
        torch.onnx.export(self.planner_net,
                          dummy_input,
                          self.onnx_save_path,
                          input_names=['input'],
                          output_names=['output']
                          )
        print("planner_net.onnx saved!")

    def load_and_test_net(self):
        planner_net_test = PlannerNet()
        planner_net_test.load_state_dict(torch.load(self.pth_save_path))
        planner_net_test.to(self.device)  # move the network to GPU

        # evaluate the network
        planner_net_test.eval()

        with torch.no_grad():
            total_loss = 0.0
            for i, data in enumerate(self.test_dataloader, 0):
                data = [d.to(self.device) for d in data]  # move data to GPU
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
