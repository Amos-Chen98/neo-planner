/*
 * @Author: Yicheng Chen (yicheng-chen@outlook.com)
 * @LastEditTime: 2023-01-24 11:22:52
 * @Modified from: https://github.com/RuPingCen/publish_pointcloud/blob/master/src/publish_pointcloud.cpp
 */

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <vector>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>

using namespace std;

int main(int argc, char **argv)
{

    std::string topic, path, frame_id;
    int hz = 5;

    ros::init(argc, argv, "publish_pointcloud");
    ros::NodeHandle nh;

    ros::param::get("~path", path);
    ros::param::get("~frame_id", frame_id);
    ros::param::get("~topic", topic);
    ros::param::get("~hz", hz);

    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>(topic, 10);

    pcl::PointCloud<pcl::PointXYZ> cloud;
    sensor_msgs::PointCloud2 output;
    pcl::io::loadPCDFile(path, cloud);
    pcl::toROSMsg(cloud, output);

    output.header.stamp = ros::Time::now();
    output.header.frame_id = frame_id;

    cout << "path = " << path << endl;
    cout << "frame_id = " << frame_id << endl;
    cout << "topic = " << topic << endl;
    cout << "hz = " << hz << endl;

    ros::Rate loop_rate(hz);
    while (ros::ok())
    {
        pcl_pub.publish(output);
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
