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
#include <pcl/filters/passthrough.h>

using namespace std;

int main(int argc, char **argv)
{

    std::string topic, path, frame_id;
    int hz;

    ros::init(argc, argv, "publish_pointcloud");
    ros::NodeHandle nh;

    ros::param::get("~path", path);
    ros::param::get("~frame_id", frame_id);
    ros::param::get("~topic", topic);
    ros::param::get("~hz", hz);

    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>(topic, 10);

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::io::loadPCDFile(path, cloud);

    // filter the point cloud
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud.makeShared());
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 15.0);
    pass.filter(cloud);

    // convert to ros message
    sensor_msgs::PointCloud2 output;
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
