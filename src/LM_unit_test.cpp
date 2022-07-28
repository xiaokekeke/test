#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "../include/utility.h"
#include "slio_sam/cloud_info.h"
#include "slio_sam/save_map.h"
#include "tinyxml.h"
using namespace std;
pcl::PointCloud<PointType> pclmap ;
pcl::PointCloud<PointType> pclscan ;
double Tms_fusion[6];//优化结束后与imu融合的结果
double Tms0[6];//优化前还未与imu数据融合的初始值
double Tms_starts[30][6];
double Tms_ends[30][6];
double chis[30];
double deltachis[30];
int coeff_count[30];
int itertotal=0;
vector<vector<vector<geometry_msgs::Point>>> all_Line_points;
void loadXML(string filename)
{
    TiXmlDocument doc;
    doc.LoadFile(filename);
    //line coeff point 需要可视化 ，或许不用自己把数据弄出来可视化也可以。
    TiXmlElement* frame = doc.FirstChildElement();
    int seq =atoi(frame->Attribute("seq"));
    cout<<"seq:"<<seq<<endl;
   
    for(TiXmlElement* elem = frame->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
    {
        string elem_name = elem->Value();
        cout<<elem_name<<endl;
        if(strcmp(elem_name.c_str(),"Tms0")==0){
            Tms0[0]=atof(elem->Attribute("rx"));
            Tms0[1]=atof(elem->Attribute("ry"));
            Tms0[2]=atof(elem->Attribute("rz"));
            Tms0[3]=atof(elem->Attribute("x"));
            Tms0[4]=atof(elem->Attribute("y"));
            Tms0[5]=atof(elem->Attribute("z"));
        }
        //pcl::pointCloud<pointType> 待会儿用publishCloud函数发送
        if(strcmp(elem_name.c_str(),"Map")==0){
             for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
                 PointType mappoint;
                 mappoint.x = atof(elem2->Attribute("x"));
                 mappoint.y =atof(elem2->Attribute("y"));
                 mappoint.z =atof(elem2->Attribute("z"));
                 mappoint.intensity =atof(elem2->Attribute("intensity"));
                 pclmap.push_back(mappoint);
             }
        }
        if(strcmp(elem_name.c_str(),"Scan")==0){
             for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
                 PointType scanpoint;
                 scanpoint.x = atof(elem2->Attribute("x"));
                 scanpoint.y =atof(elem2->Attribute("y"));
                 scanpoint.z =atof(elem2->Attribute("z"));
                 scanpoint.intensity =atof(elem2->Attribute("intensity"));
                 pclscan.push_back(scanpoint);
             }
        }
        //用于最后看融合结果对点云的影响
        if(strcmp(elem_name.c_str(),"Tms_fusion")==0){
            Tms_fusion[0]=atof(elem->Attribute("rx"));
            Tms_fusion[1]=atof(elem->Attribute("ry"));
            Tms_fusion[2]=atof(elem->Attribute("rz"));
            Tms_fusion[3]=atof(elem->Attribute("x"));
            Tms_fusion[4]=atof(elem->Attribute("y"));
            Tms_fusion[5]=atof(elem->Attribute("z"));
        }
        //这个stamp 用来给每次迭代加上时间，定时显示，当然可以用自己指定的时间戳，每帧sleep一秒
        // if(strcmp(elem_name.c_str(),"stamp")==0){
        //     cout<<elem_name <<" ";
        //     cout<<atoi(elem->Attribute("sec"))<<","<<atoi(elem->Attribute("nsec"))<<endl;
        // }
        //可视化直线和，同时里面的特征点也可以可视化，表示迭代前位置
        if(strcmp(elem_name.c_str(),"iter")==0){
            itertotal++;
            vector<vector<geometry_msgs::Point>> Line_points;
            int itercount=atoi(elem->Attribute("itercount"));
            double chi = atof(elem->Attribute("chi"));
            double deltachi = atof(elem->Attribute("deltachi"));
            //double deltachi = 0;
            double deltaR = atof(elem->Attribute("deltaR"));
            double deltaT =atof(elem->Attribute("deltaT"));
            cout<<"----------------iter:"<<itercount<<"---chi:"<<chi<<"---deltachi:"<<deltachi<<"---deltaR:"<<deltaR<<"---deltaT:"<<deltaT<<"--start-------------"<<endl;
            chis[itercount] = chi;
            deltachis[itercount] =deltachi;
          // cout<<elem_name <<" ";
           //cout<<"itercount:"<<atoi(elem->Attribute("itercount"))<<"    features_num:"<<atoi(elem->Attribute("features_num"))<<"    deltaR:"<<atof(elem->Attribute("deltaR"))<<"    deltaT"<<atof(elem->Attribute("deltaT"))<<endl;
            for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
                string elem2_name =  elem2->Value();
                if(strcmp(elem2_name.c_str(),"point_line")==0){
                        vector<geometry_msgs::Point> Line_point;
                        double s = atof(elem2->Attribute("s"));
                       
                        for(TiXmlElement* elem3 = elem2->FirstChildElement(); elem3 != NULL; elem3 = elem3->NextSiblingElement()){
                           //cout<<elem3->Value()<<" ";
                           if(strcmp(elem3->Value(),"line")==0){
                               geometry_msgs::Point point1,point2;
                               point1.x = atof(elem3->Attribute("x1"));
                               point1.y = atof(elem3->Attribute("y1"));
                               point1.z = atof(elem3->Attribute("z1"));
                               point2.x = atof(elem3->Attribute("x2"));
                               point2.y = atof(elem3->Attribute("y2"));
                               point2.z = atof(elem3->Attribute("z2"));
                               Line_point.push_back(point1);   
                               Line_point.push_back(point2);               
                           }
                           if(strcmp(elem3->Value(),"coeff")==0){
                                geometry_msgs::Point coeff;
                                double intensity = atof(elem3->Attribute("intensity"));
                                
                                coeff.x = atof(elem3->Attribute("x"))*intensity/(s*s);
                                coeff.y = atof(elem3->Attribute("y"))*intensity/(s*s);
                                coeff.z = atof(elem3->Attribute("z"))*intensity/(s*s);
                                Line_point.push_back(coeff);   
                            }
                             if(strcmp(elem3->Value(),"point")==0){
                                geometry_msgs::Point point;
                                point.x = atof(elem3->Attribute("x"));
                                point.y = atof(elem3->Attribute("y"));
                                point.z = atof(elem3->Attribute("z"));
                                Line_point.push_back(point);   
                            }
                        }
                        Line_points.push_back(Line_point);
                         ++coeff_count[itercount];
                }
                //Tms_start 是与imu融合后的结果，就是一开始的结果,我们可以利用它恢复到原始雷达帧作为和施加位移动的对比，
                if(strcmp(elem2_name.c_str(),"Tms_start")==0){
                    
                    Tms_starts[itercount][0]=atof(elem2->Attribute("rx"));
                    Tms_starts[itercount][1]=atof(elem2->Attribute("ry"));
                    Tms_starts[itercount][2]=atof(elem2->Attribute("rz"));
                    Tms_starts[itercount][3]=atof(elem2->Attribute("x"));
                    Tms_starts[itercount][4]=atof(elem2->Attribute("y"));
                    Tms_starts[itercount][5]=atof(elem2->Attribute("z"));
                    cout<<"Tms_starts:"<<Tms_starts[itercount][0]<<" "
                    <<Tms_starts[itercount][1]<<" "
                    <<Tms_starts[itercount][2]<<" "
                    <<Tms_starts[itercount][3]<<" "
                    <<Tms_starts[itercount][4]<<" "
                    <<Tms_starts[itercount][5]<<endl;
                }
                //显示每次迭代后 激光点云位置，
                if(strcmp(elem2_name.c_str(),"Tms_end")==0){
                    Tms_ends[itercount][0]=atof(elem2->Attribute("rx"));
                    Tms_ends[itercount][1]=atof(elem2->Attribute("ry"));
                    Tms_ends[itercount][2]=atof(elem2->Attribute("rz"));
                    Tms_ends[itercount][3]=atof(elem2->Attribute("x"));
                    Tms_ends[itercount][4]=atof(elem2->Attribute("y"));
                    Tms_ends[itercount][5]=atof(elem2->Attribute("z"));
                    cout<<"Tms_ends:"<<Tms_ends[itercount][0]<<" "
                    <<Tms_ends[itercount][1]<<" "
                    <<Tms_ends[itercount][2]<<" "
                    <<Tms_ends[itercount][3]<<" "
                    <<Tms_ends[itercount][4]<<" "
                    <<Tms_ends[itercount][5]<<endl;
                }
                if(strcmp(elem2_name.c_str(),"detlaX")==0){
                    cout<<"detlaX:\n"<<atof(elem2->Attribute("rx"))<<" "
                    <<atof(elem2->Attribute("ry"))<<" "
                    <<atof(elem2->Attribute("rz"))<<" "
                    <<atof(elem2->Attribute("x"))<<" "
                    <<atof(elem2->Attribute("y"))<<" "
                    <<atof(elem2->Attribute("z"))<<endl;
                }
                if(strcmp(elem2_name.c_str(),"MatE")==0){
                    cout<<"MatE:\n"<<atof(elem2->Attribute("E0"))<<" "
                    <<atof(elem2->Attribute("E1"))<<" "
                    <<atof(elem2->Attribute("E2"))<<" "
                    <<atof(elem2->Attribute("E3"))<<" "
                    <<atof(elem2->Attribute("E4"))<<" "
                    <<atof(elem2->Attribute("E5"))<<endl;
                }
                if(strcmp(elem2_name.c_str(),"MatAtB")==0){
                    cout<<"MatAtB:\n"<<atof(elem2->Attribute("M00"))<<endl
                    <<atof(elem2->Attribute("M10"))<<endl
                    <<atof(elem2->Attribute("M20"))<<endl
                    <<atof(elem2->Attribute("M30"))<<endl
                    <<atof(elem2->Attribute("M40"))<<endl
                    <<atof(elem2->Attribute("M50"))<<endl;
                }
                if(strcmp(elem2_name.c_str(),"MatH")==0){
                    cout<<"MatH:"<<endl;
                    for(int i=0;i<6;++i){
                        for(int j=0;j<6;++j){
                            cout<<atof(elem2->Attribute(("M"+to_string(i)+to_string(j)).c_str()))<<" ";
                        }
                        cout<<endl;
                    }
                }
            }

            all_Line_points.push_back(Line_points);
            cout<<"-------------------------iter:"<<itercount<<"---chi:"<<chi<<"---deltaR:"<<deltaR<<"---deltaT:"<<deltaT<<"--end---------------"<<endl;
        }
    }
    cout<<"seq:"<<seq<<endl;
    doc.Clear();
}
void transPoints(const pcl::PointCloud<PointType>& srcPoints,pcl::PointCloud<PointType>& dirPoints,double Tms[]){
    double sx=sin(Tms[0]);
    double sy=sin(Tms[1]);
    double sz=sin(Tms[2]);
    double cx=cos(Tms[0]);
    double cy=cos(Tms[1]);
    double cz=cos(Tms[2]);
    Eigen::Matrix3d Rms;
    Eigen::Vector3d tms;
    Rms<<cz*cy,cz*sy*sx-sz*cx,sz*sx+cx*cz*sy,
    sz*cy,cz*cx+sz*sy*sx,sz*sy*cx-cz*sx,
    -sy,sx*cy,cx*cy;
    tms<<Tms[3],Tms[4],Tms[5];
    for(auto point:srcPoints){
        Eigen::Vector3d dirp = Rms*Eigen::Vector3d(point.x,point.y,point.z)+tms;
        PointType dirpoint;
        dirpoint.x=dirp.x();
        dirpoint.y=dirp.y();
        dirpoint.z=dirp.z();
        dirpoint.intensity=point.intensity;
        dirPoints.push_back(dirpoint);
    }
}

void publishCloud(const ros::Publisher& thisPub, const pcl::PointCloud<PointType>& thisCloud, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(thisCloud, tempCloud);
    //tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    thisPub.publish(tempCloud);
}
void publishLMInfo(const ros::Publisher&coeffPub,const ros::Publisher&linePub,const ros::Publisher&pointPub,const int itercount,string thisFrame){
    visualization_msgs::Marker coeff_list,point_list,line_list; 
    coeff_list.header.frame_id = thisFrame;
    coeff_list.ns="coeffs";
    coeff_list.type = visualization_msgs::Marker::LINE_LIST;
    coeff_list.action = visualization_msgs::Marker::ADD;
    coeff_list.pose.orientation.w = 1.0;
    coeff_list.scale.x=0.02;
    coeff_list.color.r=0.7;
    coeff_list.color.g=0.6;
    coeff_list.color.b=0.1;
    coeff_list.color.a=0.5;
    coeff_list.points.clear();

    line_list.header.frame_id = thisFrame;
    line_list.ns="lines";
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;
    line_list.scale.x=0.02;
    line_list.color.r=0.1;
    line_list.color.g=0.6;
    line_list.color.b=0.7;
    line_list.color.a=0.7;
    line_list.points.clear();

    point_list.header.frame_id = thisFrame;
    point_list.ns = "points";
    point_list.type =visualization_msgs::Marker::POINTS;
    point_list.action = visualization_msgs::Marker::ADD;
    point_list.pose.orientation.w = 1.0;
    point_list.scale.x=0.05;
    point_list.scale.y=0.05;
    point_list.scale.z=0.05;
    point_list.color.r=0.6;
    point_list.color.g=0.1;
    point_list.color.b=0.6;
    point_list.color.a=0.8;
    point_list.points.clear();
    
    vector<vector<geometry_msgs::Point>> line_coeff_points = all_Line_points[itercount];
    for(auto line_coeff_point:line_coeff_points){
        geometry_msgs::Point coeff = line_coeff_point[2];
        geometry_msgs::Point start_p = line_coeff_point[3];
        geometry_msgs::Point end_p;
        coeff_list.points.push_back(start_p);
        end_p.x = start_p.x-coeff.x;
        end_p.y = start_p.y-coeff.y;
        end_p.z = start_p.z-coeff.z;
        //cout<<"后:"<<p<<endl;
        coeff_list.points.push_back(end_p);

        start_p = line_coeff_point[0];
        end_p  = line_coeff_point[1]; 
        line_list.points.push_back(start_p);
        line_list.points.push_back(end_p);

        point_list.points.push_back(line_coeff_point[3]);
    }
    coeffPub.publish(coeff_list);
    linePub.publish(line_list);
    pointPub.publish(point_list);

}
void se2SE(double Tms[],Eigen::Quaterniond &Qms,Eigen::Vector3d &tms){
        double sx=sin(Tms[0]);
        double sy=sin(Tms[1]);
        double sz=sin(Tms[2]);
        double cx=cos(Tms[0]);
        double cy=cos(Tms[1]);
        double cz=cos(Tms[2]);
        Eigen::Matrix3d Rms;
        Rms<<cz*cy,cz*sy*sx-sz*cx,sz*sx+cx*cz*sy,
        sz*cy,cz*cx+sz*sy*sx,sz*sy*cx-cz*sx,
        -sy,sx*cy,cx*cy;
        Qms=Eigen::Quaterniond(Rms);
        tms<<Tms[3],Tms[4],Tms[5];
}
int main(int argc, char* argv[])
{
    
    setlocale(LC_CTYPE,"zh_CN.utf8");
    
    ros::init(argc, argv, "slio_sam");
    ros::NodeHandle nh;
    //ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    ROS_INFO("\033[1;32m---->LM单元测试.\033[0m");
    string filename;
    nh.param<std::string>("/slio_sam/unit_test_path", filename, "./log/LM_log90.xml");
    loadXML(filename);
    ros::Publisher mapPublisher = nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/map",1);    
    ros::Publisher startFramePublisher = nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/startFrame",1);
    ros::Publisher start0FramePublisher = nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/start0Frame",1);//未与imu融合的结果
    ros::Publisher iterFramePublisher  =    nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/iterFrame",1);
    ros::Publisher fusionFramePublisher = nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/fusionFrame",1);
    ros::Publisher originFramePublisher =  nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/originFrame",1);
    ros::Publisher coeffPublisher = nh.advertise<visualization_msgs::Marker>("LM_unit_test/coeff",1);
    ros::Publisher linePublisher = nh.advertise<visualization_msgs::Marker>("LM_unit_test/line",1);
    ros::Publisher pointPublisher = nh.advertise<visualization_msgs::Marker>("LM_unit_test/point",1);
    ros::Publisher posePublisher = nh.advertise<geometry_msgs::PoseStamped>("/LM_unit_test/Pose",1);
    //先把map数据弄出来 ，接下来每次循环都发布
    publishCloud(mapPublisher,pclmap,"world");
    //再把原始frame弄出来，接下来巡回每次都发布
    //点云开始迭代的启始frame
    pcl::PointCloud<PointType> startframe;
    pcl::PointCloud<PointType> start0frame;
    pcl::PointCloud<PointType> fusionframe;
    pcl::PointCloud<PointType> iterframe;
    transPoints(pclscan,startframe,Tms_starts[0]);
    transPoints(pclscan,fusionframe,Tms_fusion);
    transPoints(pclscan,start0frame,Tms0);
    
    //每次迭代的Tms_start Tms_end
    //*
    
    ros::Rate loop_rate(1);//1hz
    int k=0;
    Eigen::Quaterniond qms;
    Eigen::Vector3d tms;
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id="world";
    while(ros::ok()){
        publishCloud(mapPublisher,pclmap,"world");
        publishCloud(originFramePublisher,pclscan,"world");
        publishCloud(startFramePublisher,startframe,"world");
        publishCloud(start0FramePublisher,start0frame,"world");
        publishCloud(fusionFramePublisher,fusionframe,"world");
        pcl::PointCloud<PointType> iterframe;
        se2SE(Tms_starts[k%itertotal],qms,tms);
        pose.pose.orientation.x=qms.x();
        pose.pose.orientation.y=qms.y();
        pose.pose.orientation.z=qms.z();
        pose.pose.orientation.w=qms.w();
        pose.pose.position.x=tms.x();
        pose.pose.position.y=tms.y();
        pose.pose.position.z=tms.z();
        transPoints(pclscan,iterframe,Tms_starts[k%itertotal]);
        posePublisher.publish(pose);
        ROS_INFO("iter_cout:%d  chi:%f  deltachi:%f coeff_count:%d",k%itertotal,chis[k%itertotal],deltachis[k%itertotal],coeff_count[k%itertotal]);
        //ROS_INFO("iter_cout:%d  chi:%f  deltachi:%f",k%itertotal,chis[k%itertotal],deltachis[k%itertotal]);
        publishCloud(iterFramePublisher,iterframe,"world");
        publishLMInfo(coeffPublisher,linePublisher,pointPublisher,k%itertotal,"world");
        //每次提取line coeff point 可视化 
        //每次迭代后点云位置
        ++k;
        loop_rate.sleep();
    }
   

}