#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "../include/utility.h"
#include "slio_sam/cloud_info.h"
#include "slio_sam/save_map.h"
#include <Eigen/Eigen>
#include "tinyxml.h"
using namespace std;
pcl::PointCloud<PointType> originScan;
pcl::PointCloud<PointType> deskewScan;

geometry_msgs::PoseStamped pose;
Eigen::Quaterniond Q;
Eigen::Vector3d P;
void loadXML(string filename)
{
    TiXmlDocument doc;
    doc.LoadFile(filename);
    //line coeff point 需要可视化 ，或许不用自己把数据弄出来可视化也可以。
    TiXmlElement* frame = doc.FirstChildElement();
    int seq =atoi(frame->Attribute("seq"));
    pose.header.frame_id="world";
    pose.header.seq=seq;
    cout<<"seq:"<<seq<<endl;
    for(TiXmlElement* elem = frame->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
    {
        string elem_name = elem->Value();
        cout<<elem_name<<endl;
        if(strcmp(elem_name.c_str(),"stamp")==0){
            pose.header.stamp.sec=atoi(elem->Attribute("sec"));
            pose.header.stamp.nsec=atoi(elem->Attribute("nsec"));
        }
        //pcl::pointCloud<pointType> 待会儿用publishCloud函数发送
        if(strcmp(elem_name.c_str(),"Pose")==0){
            pose.pose.orientation.x = atof(elem->Attribute("qx"));
            pose.pose.orientation.y = atof(elem->Attribute("qy"));
            pose.pose.orientation.z = atof(elem->Attribute("qz"));
            pose.pose.orientation.w = atof(elem->Attribute("qw"));
            pose.pose.position.x = atof(elem->Attribute("x"));
            pose.pose.position.y = atof(elem->Attribute("y"));
            pose.pose.position.z = atof(elem->Attribute("z"));
            Q=Eigen::Quaterniond(pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z);
            P=Eigen::Vector3d(pose.pose.position.x,pose.pose.position.y,pose.pose.position.z);
        }
        if(strcmp(elem_name.c_str(),"orignal_scan")==0){
             for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
                 PointType mappoint;
                 mappoint.x = atof(elem2->Attribute("x"));
                 mappoint.y =atof(elem2->Attribute("y"));
                 mappoint.z =atof(elem2->Attribute("z"));
                 mappoint.intensity =atof(elem2->Attribute("intensity"));
                 originScan.push_back(mappoint);
             }
        }
        if(strcmp(elem_name.c_str(),"deskew_scan")==0){
             for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
                 PointType scanpoint;
                 scanpoint.x = atof(elem2->Attribute("x"));
                 scanpoint.y =atof(elem2->Attribute("y"));
                 scanpoint.z =atof(elem2->Attribute("z"));
                 scanpoint.intensity =atof(elem2->Attribute("intensity"));
                 deskewScan.push_back(scanpoint);
             }
        }
    }
    cout<<"seq:"<<seq<<endl;
    doc.Clear();
}
void transPoints(const pcl::PointCloud<PointType>& thisCloud,pcl::PointCloud<PointType>& returnCloud,const Eigen::Quaterniond& rotation,const Eigen::Vector3d& position){
    PointType dirpoint;
    for(auto srcpoint : thisCloud){
        Eigen::Vector3d vpoint(srcpoint.x,srcpoint.y,srcpoint.z);
        Eigen::Vector3d avpoint=rotation*vpoint+position;
        dirpoint.x=avpoint.x();
        dirpoint.y=avpoint.y();
        dirpoint.z=avpoint.z();
        dirpoint.intensity = srcpoint.intensity;
        returnCloud.push_back(dirpoint);
    }
}
void publishCloud(const ros::Publisher& thisPub, const pcl::PointCloud<PointType>& thisCloud, std::string thisFrame){
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(thisCloud, tempCloud);
    //tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    thisPub.publish(tempCloud);
}
int main(int argc, char* argv[])
{
    
    setlocale(LC_CTYPE,"zh_CN.utf8");
    
    ros::init(argc, argv, "slio_sam");
    ros::NodeHandle nh;
    //ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    ROS_INFO("\033[1;32m---->点云去畸变单元测试.\033[0m");
    string filename;
    nh.param<std::string>("/slio_sam/unit_test_path", filename, "./log/deskew_log90.xml");
    loadXML(filename);

    ros::Publisher deskewFramePublisher = nh.advertise<sensor_msgs::PointCloud2>("/deskew_unit_test/deskewFrame",1);

    ros::Publisher originFramePublisher =  nh.advertise<sensor_msgs::PointCloud2>("/deskew_unit_test/originFrame",1);

    ros::Publisher posePublisher = nh.advertise<geometry_msgs::PoseStamped>("/deskew_unit_test/Pose",1);
    //先把map数据弄出来 ，接下来每次循环都发布
    //再把原始frame弄出来，接下来巡回每次都发布
    //点云开始迭代的启始frame    
    //每次迭代的Tms_start Tms_end
    //*
    
    ros::Rate loop_rate(10);//1hz
    pcl::PointCloud<PointType> originframe,deskewframe;
    transPoints(originScan,originframe,Q,P);
    transPoints(deskewScan,deskewframe,Q,P);
    while(ros::ok()){
        publishCloud(originFramePublisher,originframe,"world");
        publishCloud(deskewFramePublisher,deskewframe,"world");
        posePublisher.publish(pose);
        loop_rate.sleep();
    }
   

}