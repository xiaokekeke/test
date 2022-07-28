#include "sensor_msgs/LaserScan.h"
#include <ros/ros.h>
#include<iostream>
using namespace std;
sensor_msgs::LaserScan scan;
double scanhz=0,scan_sample=0;
void sim_scan_change(const::sensor_msgs::LaserScanConstPtr &msg){
    scan = *msg;
    scan.scan_time=1.0/scanhz;
    scan.time_increment=(1.0/scanhz)/scan_sample;
}
int main (int argc, char** argv){
    setlocale(LC_CTYPE,"zh_CN.utf8");
    ros::init(argc, argv, "slio_sam");
    ros::NodeHandle nh;
    nh.param<double>("/scanHZ",scanhz, 15);
    nh.param<double>("/scanSample",scan_sample, 1443);
    ros::Subscriber sim_scan_sub = nh.subscribe<sensor_msgs::LaserScan>("sim_scan", 1,sim_scan_change);
    ros::Publisher scan_pub = nh.advertise<sensor_msgs::LaserScan>("scan",1);
    ros::Rate loop_rate(15);
    while(ros::ok()){
        scan_pub.publish(scan);
        ros::spinOnce();
        loop_rate.sleep();
    }
   
    return 0;
}
