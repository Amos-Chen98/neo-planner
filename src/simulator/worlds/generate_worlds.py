'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2024-02-26 10:50:36
This file has no connection to ROS. It is used to generate gazebo worlds.
'''

import os
import xml.etree.ElementTree as ET
import copy
import numpy as np


class Model():
    def __init__(self):
        self.name = None
        self.size = None
        self.pose = None


class GeneratorConfig:
    def __init__(self):
        # -------------------customized parameters-------------------
        template_world_name = 'poles.world'

        self.new_model_num = [5, 10, 15, 20, 30, 40]  # the number of models in each new world, can be regarded as the density of obstacles
        self.new_model_pose_range = [3, 27, -5, 5]  # [x_min, x_max, y_min, y_max
        self.new_model_size_range = [0.5, 1.5, 0.5, 1.5, 3, 6]  # [x_min, x_max, y_min, y_max, z_min, z_max]

        # -------------------end of customized parameters-------------------

        self.new_world_num = len(self.new_model_num)  # the number of new worlds to be generated

        # create a list of world names. Each name is determinded by the number of models in the world
        self.new_world_names = ['rand_world_'+str(num) for num in self.new_model_num]
        new_world_names_ref = copy.deepcopy(self.new_world_names)

        for i in range(len(new_world_names_ref)):
            # if multiple worlds have the same number of models, add a suffix to the name
            if new_world_names_ref.count(new_world_names_ref[i]) > 1:
                self.new_world_names[i] += '(' + str(new_world_names_ref[0:i].count(new_world_names_ref[i]) + 1) + ')'

        self.template_world_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), template_world_name)
        self.new_world_path_list = [os.path.join(os.path.abspath(os.path.dirname(__file__)), name + '.world') for name in self.new_world_names]


class WorldGenerator:
    def __init__(self, config=GeneratorConfig()):
        self.config = config

        self.init_template_world()  # clear all existing models (obstacles, not including the ground plane)

    def init_template_world(self):

        self.tree = ET.parse(self.config.template_world_path)
        self.root = self.tree.getroot()  # the root is sdf
        self.world = self.root.find('world')  # sdf > world
        self.state = self.world.find('state')  # sdf > world > state

        self.all_world_models = self.world.findall('model')
        self.all_state_models = self.state.findall('model')

        self.template_world_model = self.all_world_models[1]  # [0] is the ground_plane
        self.template_state_model = self.all_state_models[1]  # [0] is the ground_plane

        # Remove all existing models in the template world
        for model in self.all_world_models:
            model_name = model.attrib['name']
            if model_name != 'ground_plane':
                self.world.remove(model)

        for model in self.all_state_models:
            model_name = model.attrib['name']
            if model_name != 'ground_plane':
                self.state.remove(model)

    def generate_models(self, model_num):
        '''
        Generate new models with random size and pose
        '''
        self.new_models = []

        # print("new_model_num: %d\n" % model_num)

        for i in range(model_num):
            new_model = Model()
            new_model.name = 'model' + str(i)

            # get model size
            model_size_x_range = self.config.new_model_size_range[1] - self.config.new_model_size_range[0]
            model_size_y_range = self.config.new_model_size_range[3] - self.config.new_model_size_range[2]
            model_size_z_range = self.config.new_model_size_range[5] - self.config.new_model_size_range[4]
            model_size_x = np.random.rand() * model_size_x_range + self.config.new_model_size_range[0]
            model_size_y = np.random.rand() * model_size_y_range + self.config.new_model_size_range[2]
            model_size_z = np.random.rand() * model_size_z_range + self.config.new_model_size_range[4]

            # get model pose
            model_pose_x_range = self.config.new_model_pose_range[1] - self.config.new_model_pose_range[0]
            model_pose_y_range = self.config.new_model_pose_range[3] - self.config.new_model_pose_range[2]
            model_pose_x = np.random.rand() * model_pose_x_range + self.config.new_model_pose_range[0]
            model_pose_y = np.random.rand() * model_pose_y_range + self.config.new_model_pose_range[2]
            model_pose_z = model_size_z / 2

            # set model size and pose
            new_model.size = np.array([model_size_x, model_size_y, model_size_z])
            new_model.pose = np.array([model_pose_x, model_pose_y, model_pose_z, 0, 0, 0])

            # Determine if the position of the model conflicts with existing models
            while True:
                conflict = False
                for j in range(i):
                    if abs(new_model.pose[0] - self.new_models[j].pose[0]) < (new_model.size[0] + self.new_models[j].size[0]) / 2 and abs(new_model.pose[1] - self.new_models[j].pose[1]) < (new_model.size[1] + self.new_models[j].size[1]) / 2:
                        conflict = True
                        break
                if conflict:
                    model_pose_x = np.random.rand() * model_pose_x_range + self.config.new_model_pose_range[0]
                    model_pose_y = np.random.rand() * model_pose_y_range + self.config.new_model_pose_range[2]
                    model_pose_z = model_size_z / 2
                    new_model.pose = np.array([model_pose_x, model_pose_y, model_pose_z, 0, 0, 0])
                else:
                    break

            # print("---------model", i, "---------")
            # print("Size:", new_model.size)
            # print("Pose:", new_model.pose)
            # print("")

            self.new_models.append(new_model)

    def generate_world(self, world_index):
        self.init_template_world()
        model_num = self.config.new_model_num[world_index]
        world_path = self.config.new_world_path_list[world_index]
        self.generate_models(model_num)

        for i in range(model_num):
            self.add_world_model(self.new_models[i].size, self.new_models[i].name)
            self.add_state_model(self.new_models[i].pose, self.new_models[i].name)

        world_generator.save_world(world_path)

    def batch_generate_world(self):
        for i in range(self.config.new_world_num):
            self.generate_world(i)

    def add_world_model(self, model_size_value, model_name):
        '''
        Add a new model as the world's child
        '''
        new_world_model = copy.deepcopy(self.template_world_model)
        new_world_model.attrib['name'] = model_name
        link = new_world_model.find('link')
        visual = link.find('visual')
        visual_geometry = visual.find('geometry')
        visual_box = visual_geometry.find('box')
        visual_box_size = visual_box.find('size')

        collision = link.find('collision')
        collision_geometry = collision.find('geometry')
        collision_box = collision_geometry.find('box')
        collision_size = collision_box.find('size')

        model_size_text = ' '.join([str(i) for i in model_size_value])
        visual_box_size.text = model_size_text
        collision_size.text = model_size_text

        self.world.append(new_world_model)

    def add_state_model(self, model_pose_value, model_name):
        '''
        Add a new model as the state's child
        '''
        new_state_model = copy.deepcopy(self.template_state_model)
        new_state_model.attrib['name'] = model_name
        model_pose = new_state_model.find('pose')
        link = new_state_model.find('link')
        link_pose = link.find('pose')

        model_pose_text = ' '.join([str(i) for i in model_pose_value])
        link_pose.text = model_pose_text
        model_pose.text = model_pose_text

        self.state.append(new_state_model)

    def save_world(self, world_path):

        self.tree.write(world_path, 'UTF-8')

        print("New world saved at:", world_path)


if __name__ == "__main__":

    world_generator = WorldGenerator()

    world_generator.batch_generate_world()
