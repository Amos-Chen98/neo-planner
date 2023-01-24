/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin_build_octomap.h"

#include <octomap_msgs/conversions.h>
#include <gazebo/common/Time.hh>
#include <gazebo/common/CommonTypes.hh>
#include <ignition/math/Vector3.hh>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include <octomap/octomap.h>
#include <octomap/octomap_timing.h>

using namespace octomap;
using namespace std;

namespace gazebo
{

    OctomapFromGazeboWorld::~OctomapFromGazeboWorld()
    {
        delete octomap_;
        octomap_ = NULL;
    }

    void OctomapFromGazeboWorld::Load(physics::WorldPtr _parent,
                                      sdf::ElementPtr _sdf)
    {
        if (kPrintOnPluginLoad)
        {
            gzdbg << __FUNCTION__ << "() called." << std::endl;
        }

        world_ = _parent;

        std::string service_name = "world/build_octomap";
        std::string octomap_pub_topic = "world/octomap";
        getSdfParam<std::string>(_sdf, "octomapPubTopic", octomap_pub_topic,
                                 octomap_pub_topic);
        getSdfParam<std::string>(_sdf, "octomapServiceName", service_name,
                                 service_name);

        gzlog << "Advertising service: " << service_name << std::endl;
        srv_ = node_handle_.advertiseService(
            service_name, &OctomapFromGazeboWorld::ServiceCallback, this);
        octomap_publisher_ =
            node_handle_.advertise<octomap_msgs::Octomap>(octomap_pub_topic, 1, true);
    }

    bool OctomapFromGazeboWorld::ServiceCallback(
        sim_gazebo_plugins::Octomap::Request &req, sim_gazebo_plugins::Octomap::Response &res)
    {
        gzlog << "Creating octomap with origin at (" << req.bounding_box_origin.x
              << ", " << req.bounding_box_origin.y << ", "
              << req.bounding_box_origin.z << "), and bounding box lengths ("
              << req.bounding_box_lengths.x << ", " << req.bounding_box_lengths.y
              << ", " << req.bounding_box_lengths.z
              << "), and leaf size: " << req.leaf_size << ".\n";
        CreateOctomap(req);

        unsigned int maxDepth = octomap_->getTreeDepth();
        cout << "tree depth is " << maxDepth << endl;

        // expand collapsed occupied nodes until all occupied leaves are at maximum depth
        vector<OcTreeNode *> collapsed_occ_nodes;
        do
        {
            collapsed_occ_nodes.clear();
            for (OcTree::iterator it = octomap_->begin(); it != octomap_->end(); ++it)
            {
                if (octomap_->isNodeOccupied(*it) && it.getDepth() < maxDepth)
                {
                    collapsed_occ_nodes.push_back(&(*it));
                }
            }
            for (vector<OcTreeNode *>::iterator it = collapsed_occ_nodes.begin(); it != collapsed_occ_nodes.end(); ++it)
            {
                octomap_->expandNode(*it);
            }
            cout << "expanded " << collapsed_occ_nodes.size() << " nodes" << endl;
        } while (collapsed_occ_nodes.size() > 0);

        vector<point3d> pcl;
        for (OcTree::iterator it = octomap_->begin(); it != octomap_->end(); ++it)
        {
            if (octomap_->isNodeOccupied(*it))
            {
                pcl.push_back(it.getCoordinate());
            }
        }
        std::string ex = req.filename;
        std::string toReplace(".bt");
        size_t pos = ex.find(toReplace);
        ex.replace(pos, toReplace.length(), "");
        string outputFilename = ex + ".pcd";
        ofstream f(outputFilename.c_str(), ofstream::out);
        f << "# .PCD v0.7" << endl
          << "VERSION 0.7" << endl
          << "FIELDS x y z" << endl
          << "SIZE 4 4 4" << endl
          << "TYPE F F F" << endl
          << "COUNT 1 1 1" << endl
          << "WIDTH " << pcl.size() << endl
          << "HEIGHT 1" << endl
          << "VIEWPOINT 0 0 0 0 0 0 1" << endl
          << "POINTS " << pcl.size() << endl
          << "DATA ascii" << endl;
        for (size_t i = 0; i < pcl.size(); i++)
            f << pcl[i].x() << " " << pcl[i].y() << " " << pcl[i].z() << endl;
        f.close();

        if (req.filename != "")
        {
            if (octomap_)
            {
                std::string path = req.filename;
                octomap_->writeBinary(path);
                gzlog << std::endl
                      << "Octree saved as " << path << std::endl;
            }
            else
            {
                ROS_ERROR("The octree is NULL. Will not save that.");
            }
        }

        common::Time now = world_->SimTime();
        res.map.header.frame_id = "world";
        res.map.header.stamp = ros::Time(now.sec, now.nsec);

        if (!octomap_msgs::binaryMapToMsg(*octomap_, res.map))
        {
            ROS_ERROR("Error serializing OctoMap");
        }

        if (req.publish_octomap)
        {
            gzlog << "Publishing Octomap." << std::endl;
            octomap_publisher_.publish(res.map);
        }

        common::SphericalCoordinatesPtr sphericalCoordinates = world_->SphericalCoords();
        ignition::math::Vector3d origin_cartesian(0.0, 0.0, 0.0);
        ignition::math::Vector3d origin_spherical = sphericalCoordinates->SphericalFromLocal(origin_cartesian);

        res.origin_latitude = origin_spherical.X();
        res.origin_longitude = origin_spherical.Y();
        res.origin_altitude = origin_spherical.Z();
        return true;
    }

    void OctomapFromGazeboWorld::FloodFill(
        const ignition::math::Vector3d &seed_point, const ignition::math::Vector3d &bounding_box_origin,
        const ignition::math::Vector3d &bounding_box_lengths, const double leaf_size)
    {
        octomap::OcTreeNode *seed =
            octomap_->search(seed_point[0], seed_point[1], seed_point[2]);
        // do nothing if point occupied
        if (seed != NULL && seed->getOccupancy())
            return;

        std::stack<octomath::Vector3> to_check;
        to_check.push(octomath::Vector3(seed_point[0], seed_point[1], seed_point[2]));

        while (to_check.size() > 0)
        {
            octomath::Vector3 p = to_check.top();

            if ((p.x() > bounding_box_origin[0] - bounding_box_lengths[0] / 2) &&
                (p.x() < bounding_box_origin[0] + bounding_box_lengths[0] / 2) &&
                (p.y() > bounding_box_origin[1] - bounding_box_lengths[1] / 2) &&
                (p.y() < bounding_box_origin[1] + bounding_box_lengths[1] / 2) &&
                (p.z() > bounding_box_origin[2] - bounding_box_lengths[2] / 2) &&
                (p.z() < bounding_box_origin[2] + bounding_box_lengths[2] / 2) &&
                (!octomap_->search(p)))
            {
                octomap_->setNodeValue(p, 0);
                to_check.pop();
                to_check.push(octomath::Vector3(p.x() + leaf_size, p.y(), p.z()));
                to_check.push(octomath::Vector3(p.x() - leaf_size, p.y(), p.z()));
                to_check.push(octomath::Vector3(p.x(), p.y() + leaf_size, p.z()));
                to_check.push(octomath::Vector3(p.x(), p.y() - leaf_size, p.z()));
                to_check.push(octomath::Vector3(p.x(), p.y(), p.z() + leaf_size));
                to_check.push(octomath::Vector3(p.x(), p.y(), p.z() - leaf_size));
            }
            else
            {
                to_check.pop();
            }
        }
    }

    bool OctomapFromGazeboWorld::CheckIfInterest(const ignition::math::Vector3d &central_point,
                                                 gazebo::physics::RayShapePtr ray,
                                                 const double leaf_size)
    {
        ignition::math::Vector3d start_point = central_point;
        ignition::math::Vector3d end_point = central_point;

        double dist;
        std::string entity_name;

        start_point[0] += leaf_size / 2;
        end_point[0] -= leaf_size / 2;
        ray->SetPoints(start_point, end_point);
        ray->GetIntersection(dist, entity_name);

        if (dist <= leaf_size)
            return true;

        start_point = central_point;
        end_point = central_point;
        start_point[1] += leaf_size / 2;
        end_point[1] -= leaf_size / 2;
        ray->SetPoints(start_point, end_point);
        ray->GetIntersection(dist, entity_name);

        if (dist <= leaf_size)
            return true;

        start_point = central_point;
        end_point = central_point;
        start_point[2] += leaf_size / 2;
        end_point[2] -= leaf_size / 2;
        ray->SetPoints(start_point, end_point);
        ray->GetIntersection(dist, entity_name);

        if (dist <= leaf_size)
            return true;

        return false;
    }

    void OctomapFromGazeboWorld::CreateOctomap(
        const sim_gazebo_plugins::Octomap::Request &msg)
    {
        const double epsilon = 0.00001;
        const int far_away = 100000;
        ignition::math::Vector3d bounding_box_origin(msg.bounding_box_origin.x,
                                                     msg.bounding_box_origin.y,
                                                     msg.bounding_box_origin.z);
        // epsilion prevents undefiened behaviour if a point is inserted exactly
        // between two octomap cells
        ignition::math::Vector3d bounding_box_lengths(msg.bounding_box_lengths.x + epsilon,
                                                      msg.bounding_box_lengths.y + epsilon,
                                                      msg.bounding_box_lengths.z + epsilon);
        double leaf_size = msg.leaf_size;
        octomap_ = new octomap::OcTree(leaf_size);
        octomap_->clear();
        octomap_->setProbHit(0.7);
        octomap_->setProbMiss(0.4);
        octomap_->setClampingThresMin(0.12);
        octomap_->setClampingThresMax(0.97);
        octomap_->setOccupancyThres(0.7);

        gazebo::physics::PhysicsEnginePtr engine = world_->Physics();
        engine->InitForThread();
        gazebo::physics::RayShapePtr ray =
            boost::dynamic_pointer_cast<gazebo::physics::RayShape>(
                engine->CreateShape("ray", gazebo::physics::CollisionPtr()));

        std::cout << "Rasterizing world and checking collisions" << std::endl;

        for (double x =
                 leaf_size / 2 + bounding_box_origin[0] - bounding_box_lengths[0] / 2;
             x < bounding_box_origin[0] + bounding_box_lengths[0] / 2; x += leaf_size)
        {
            int progress =
                round(100 * (x + bounding_box_lengths[0] / 2 - bounding_box_origin[0]) /
                      bounding_box_lengths[0]);
            std::cout << "\rPlacing model edges into octomap... " << progress
                      << "%                 ";

            for (double y =
                     leaf_size / 2 + bounding_box_origin[1] - bounding_box_lengths[1] / 2;
                 y < bounding_box_origin[1] + bounding_box_lengths[1] / 2;
                 y += leaf_size)
            {
                for (double z = leaf_size / 2 + bounding_box_origin[2] -
                                bounding_box_lengths[2] / 2;
                     z < bounding_box_origin[2] + bounding_box_lengths[2] / 2;
                     z += leaf_size)
                {
                    ignition::math::Vector3d point(x, y, z);
                    if (CheckIfInterest(point, ray, leaf_size))
                    {
                        octomap_->setNodeValue(x, y, z, 1);
                    }
                }
            }
        }
        octomap_->prune();
        octomap_->updateInnerOccupancy();

        // flood fill from top and bottom
        std::cout << "\rFlood filling freespace...                                  ";
        FloodFill(ignition::math::Vector3d(bounding_box_origin[0] + leaf_size / 2,
                                           bounding_box_origin[1] + leaf_size / 2,
                                           bounding_box_origin[2] + bounding_box_lengths[2] / 2 -
                                               leaf_size / 2),
                  bounding_box_origin, bounding_box_lengths, leaf_size);
        FloodFill(ignition::math::Vector3d(bounding_box_origin[0] + leaf_size / 2,
                                           bounding_box_origin[1] + leaf_size / 2,
                                           bounding_box_origin[2] - bounding_box_lengths[2] / 2 +
                                               leaf_size / 2),
                  bounding_box_origin, bounding_box_lengths, leaf_size);

        octomap_->prune();
        octomap_->updateInnerOccupancy();

        // set unknown to filled
        for (double x =
                 leaf_size / 2 + bounding_box_origin[0] - bounding_box_lengths[0] / 2;
             x < bounding_box_origin[0] + bounding_box_lengths[0] / 2; x += leaf_size)
        {
            int progress =
                round(100 * (x + bounding_box_lengths[0] / 2 - bounding_box_origin[0]) /
                      bounding_box_lengths[0]);
            std::cout << "\rFilling closed spaces... " << progress << "%              ";

            for (double y =
                     leaf_size / 2 + bounding_box_origin[1] - bounding_box_lengths[1] / 2;
                 y < bounding_box_origin[1] + bounding_box_lengths[1] / 2;
                 y += leaf_size)
            {
                for (double z = leaf_size / 2 + bounding_box_origin[2] -
                                bounding_box_lengths[2] / 2;
                     z < bounding_box_origin[2] + bounding_box_lengths[2] / 2;
                     z += leaf_size)
                {
                    octomap::OcTreeNode *seed = octomap_->search(x, y, z);
                    if (!seed)
                        octomap_->setNodeValue(x, y, z, 1);
                }
            }
        }

        octomap_->prune();
        octomap_->updateInnerOccupancy();

        std::cout << "\rOctomap generation completed                  " << std::endl;
    }

    // Register this plugin with the simulator
    GZ_REGISTER_WORLD_PLUGIN(OctomapFromGazeboWorld)

} // namespace gazebo
