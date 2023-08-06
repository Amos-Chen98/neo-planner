'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-08-06 21:59:59
'''
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import time
import torch.onnx
import torchvision
import torchvision.models as models
import cv2
import onnx
import onnxruntime


IMG_WIDTH = 480
IMG_HEIGHT = 360
VECTOR_SIZE = 24
OUTPUT_SIZE = 9
BATCH_SIZE = 64
EPOCHS = 30

current_path = os.path.dirname(os.path.abspath(__file__))[:-8]  # -7 remove '/scripts'
img_path = '/training_data/starred/depth_img'
csv_path = '/training_data/starred/train.csv'
pth_save_path = '/saved_net/planner_net.pth'
onnx_save_path = '/saved_net/planner_net.onnx'
trt_save_path = '/saved_net/planner_net.trt'


def process_input_np(depth_img, motion_info):
    '''
    :param depth_img: numpy array
    :param motion_info: numpy array
    :return: numpy array
    '''
    depth_img_resized = cv2.resize(depth_img, (IMG_WIDTH, IMG_HEIGHT))
    img_flatten = depth_img_resized.reshape(-1)
    return np.concatenate((img_flatten.astype(np.float32), motion_info.astype(np.float32)))


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

            # form input
            img_file_name = os.path.join(self.img_path, f'{timestamp}.png')
            depth_img = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
            input_np = process_input_np(depth_img, row[1:-9].values)
            input = torch.tensor(input_np.astype(np.float32))

            # form output
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

        self.img_backbone = models.resnet18(weights='DEFAULT')

        for param in self.img_backbone.parameters():
            # freeze the parameters
            param.requires_grad = False

        # change the input channel to 1
        self.img_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change the output size to fit VECTOR_SIZE, self.img_backbone.fc.in_features means the layer's original input size
        self.img_backbone.fc = nn.Linear(self.img_backbone.fc.in_features, VECTOR_SIZE)

        self.motion_backbone = nn.Sequential(
            nn.Linear(VECTOR_SIZE, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 24),
            nn.LeakyReLU(),
            nn.Linear(24, VECTOR_SIZE)
        )

        self.mlp = nn.Sequential(
            nn.Linear(VECTOR_SIZE + VECTOR_SIZE, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 96),
            nn.LeakyReLU(),
            nn.Linear(96, OUTPUT_SIZE)
        )

    def forward(self, input):
        # retrieve the image and vector from the input
        img = input[:, :IMG_WIDTH * IMG_HEIGHT].reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT)
        vector = input[:, IMG_WIDTH * IMG_HEIGHT:]

        img_feature = self.img_backbone(img)
        motion_feature = self.motion_backbone(vector)

        x = torch.cat([img_feature, motion_feature], dim=1)
        x = self.mlp(x)

        return x


class NNTrainer():
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

    def train_net(self):

        self.build_dataset()
        self.init_net()
        optimizer = optim.Adam(self.planner_net.parameters(), lr=0.001)

        print("Start training...")
        time_start = time.time()

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

        time_end = time.time()
        training_time = time_end - time_start

        # convert the training time to hour:minute:second
        m, s = divmod(training_time, 60)
        h, m = divmod(m, 60)
        print("\nTraining time cost: %d:%02d:%02d\n" % (h, m, s))

    def save_test_pth_model(self):
        torch.save(self.planner_net.state_dict(), self.pth_save_path)
        print("pth model saved!")

        self.test_pth_model()

    def test_pth_model(self):
        pth_model = PlannerNet()
        pth_model.load_state_dict(torch.load(self.pth_save_path))
        pth_model.to(self.device)  # move the network to GPU

        # evaluate the network
        pth_model.eval()

        with torch.no_grad():
            total_loss = 0.0
            for i, data in enumerate(self.test_dataloader, 0):
                data = [d.to(self.device) for d in data]  # move data to GPU
                inputs, expert_outputs = data
                outputs = pth_model(inputs)
                loss = self.criterion(outputs, expert_outputs)
                total_loss += loss.item()

            print('Test loss of pth model: %.3f' % (total_loss / len(self.test_dataloader)))

    def save_test_onnx_model(self):
        dummy_input = torch.randn(1, IMG_WIDTH*IMG_HEIGHT+VECTOR_SIZE)
        dummy_input = dummy_input.to(self.device)  # move the dummy input to GPU
        torch.onnx.export(self.planner_net,
                          dummy_input,
                          self.onnx_save_path,
                          input_names=['input'],
                          output_names=['output']
                          )
        print("onnx model saved!")

        self.test_onnx_model()

    def test_onnx_model(self):
        # load the onnx model
        onnx_model = onnx.load(self.onnx_save_path)
        onnx.checker.check_model(onnx_model)
        print("onnx model checked.")

    def generate_a_random_input(self):
        img = torch.randn(1, 1, IMG_WIDTH, IMG_HEIGHT)
        vector = torch.randn(1, VECTOR_SIZE)


if __name__ == '__main__':

    nn_trainer = NNTrainer()

    nn_trainer.train_net()

    nn_trainer.save_test_pth_model()

    nn_trainer.save_test_onnx_model()
