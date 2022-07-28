#include "../include/utility.h"
#include "slio_sam/cloud_info.h"
#include "slio_sam/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include<fstream>
#include <gtsam/nonlinear/ISAM2.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include "tinyxml.h"
#define XMLLOG 0
#define PROCESS 0
#define TIME_TEST 0
#define LM2 1
#define LM1 0
using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;
    visualization_msgs::Marker line_list;//这个是自己加的主要是为了显示提取的直线正确与否
    visualization_msgs::Marker coeff_list;
    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;
    ros::Publisher marker_pub;//这个是自己加的主要是为了显示提取的直线正确与否
    ros::Publisher coeff_pub;
    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;
    //std::ofstream logname;
#if XMLLOG
    TiXmlDocument *doc;//存放日志文件，用xml 格式
    TiXmlElement *frame;
    TiXmlElement *stamp;
    TiXmlElement *iter;
#endif

    ros::Publisher pubSLAMInfo;
    
    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    slio_sam::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;
    pcl::PointCloud<PointType> savelaserCloudOri;
    pcl::PointCloud<PointType> savecoeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    //pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;
    double lastchi,chi,theta;
    float transformTobeMapped[6];
    float transformTobeMapped_last[6];
    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    string convertToString(double d) {
	ostringstream os;
	    if (os << d)
		    return os.str();
	    return "invalid conversion";
    }
    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);
        //
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("slio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("slio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("slio_sam/mapping/path", 1);
        marker_pub             =nh.advertise<visualization_msgs::Marker>("slio_sam/mapping/line_extract",10);
        coeff_pub                  =nh.advertise<visualization_msgs::Marker>("slio_sam/mapping/coeff",10);
        subCloud                   = nh.subscribe<slio_sam::cloud_info>("slio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        //subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        //subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        srvSaveMap             = nh.advertiseService("slio_sam/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/slio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/mapping/cloud_registered_raw", 1);

        pubSLAMInfo     = nh.advertise<slio_sam::cloud_info>("slio_sam/mapping/slam_info", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        //ROS_INFO("allocateMemory()_内存分配开始");
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        //kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
        //ROS_INFO("allocateMemory()_内存分配结束");
    }

    void laserCloudInfoHandler(const slio_sam::cloud_infoConstPtr& msgIn)
    {
         // 当前激光帧时间戳
        timeLaserInfoStamp = msgIn->header.stamp;
        
        timeLaserInfoCur = msgIn->header.stamp.toSec();
        //发布直线
        line_list.header.stamp=msgIn->header.stamp;
        coeff_list.header.stamp=msgIn->header.stamp;


        //发布直线end
       // 提取当前激光帧角点、平面点集合
        cloudInfo = *msgIn;
        //ROS_INFO("执行转换");
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);//将sensor_msgs::PointCloud2转化为pcl::PointCloud
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
        //ROS_INFO("mapOptmization 角特征点数量:%d,面特征点数量:%d",laserCloudCornerLast->size(),laserCloudSurfLast->size());
        std::lock_guard<std::mutex> lock(mtx);
         // mapping执行频率控制
        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
 #if    XMLLOG          
            int seq=msgIn->header.seq;
            //logname.open("/home/shen/USTC/all_SLAM/slio_sam_ws/log/LM_log"+to_string(seq)+".txt");

            frame = new TiXmlElement("frame");
            doc = new TiXmlDocument();
            doc->LinkEndChild(frame);
            frame->SetAttribute("seq",to_string(seq));
            stamp = new TiXmlElement("stamp");
            frame->LinkEndChild(stamp);
            stamp->SetAttribute("sec",to_string(timeLaserInfoStamp.sec));
            stamp->SetAttribute("nsec",to_string(timeLaserInfoStamp.nsec));
#endif    
            //logname<<to_string(seq)<<"帧"<<endl;
            //logname<<"Stamp:"<<timeLaserInfoStamp<<"\n"<<endl;
            timeLastProcessing = timeLaserInfoCur;
            
            // 当前帧位姿初始化
             // 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
              // 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
            //ROS_INFO("------------------------优化前准备工作-start-------------------------");
 #if   XMLLOG   
            TiXmlElement *Tms0 = new TiXmlElement("Tms0");//进入优化和imu融合前
            frame->LinkEndChild(Tms0);
            Tms0->SetAttribute("rx",convertToString(transformTobeMapped[0]));
            Tms0->SetAttribute("ry",convertToString(transformTobeMapped[1]));
            Tms0->SetAttribute("rz",convertToString(transformTobeMapped[2]));
            Tms0->SetAttribute("x",convertToString(transformTobeMapped[3]));
            Tms0->SetAttribute("y",convertToString(transformTobeMapped[4]));
            Tms0->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif 
            // double transformTobeMapped_old[6];
            // transformTobeMapped_old[0]=transformTobeMapped[0];
            // transformTobeMapped_old[1]=transformTobeMapped[1];
            // transformTobeMapped_old[2]=transformTobeMapped[2];
            // transformTobeMapped_old[3]=transformTobeMapped[3];
            // transformTobeMapped_old[4]=transformTobeMapped[4];
            // transformTobeMapped_old[5]=transformTobeMapped[5];

            // //记录和imu融合前的结果
            updateInitialGuess();
            // double alterR,alterT;
            // alterR = std::abs(transformTobeMapped_old[2]-transformTobeMapped[2]);
            // alterT = std::abs(transformTobeMapped_old[3]-transformTobeMapped[3])+std::abs(transformTobeMapped_old[4]-transformTobeMapped[4]);
            // if(alterR>0.15||alterT>0.3){
            //     ROS_INFO("IMU数据有问题,融合前后,差别过大");
            //     transformTobeMapped[0]=transformTobeMapped_old[0];
            //     transformTobeMapped[1]=transformTobeMapped_old[1];
            //     transformTobeMapped[2]=transformTobeMapped_old[2];
            //     transformTobeMapped[3]=transformTobeMapped_old[3];
            //     transformTobeMapped[4]=transformTobeMapped_old[4];
            //     transformTobeMapped[5]=transformTobeMapped_old[5];
            // }
             // 提取局部角点、平面点云集合，加入局部map
             // 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
              // 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
            extractSurroundingKeyFrames();
           // 当前激光帧角点、平面点集合降采样
            downsampleCurrentScan();
           // ROS_INFO("------------------------优化前准备工作-end-------------------------");
            // scan-to-map优化当前帧位姿
            // 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
            // 2、迭代30次（上限）优化
            //    1) 当前激光帧角点寻找局部map匹配点
            //       a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
            //       b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
            //    2) 当前激光帧平面点寻找局部map匹配点
            //       a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
            //       b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
            //    3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
            //    4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
            // 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    
            scan2MapOptimization();
            // 设置当前帧为关键帧并执行因子图优化
            // 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
            // 2、添加激光里程计因子、GPS因子、闭环因子
            // 3、执行因子图优化
            // 4、得到当前帧优化后位姿，位姿协方差
            // 5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
            saveKeyFramesAndFactor();
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
            correctPoses();
            
            //发布里程记
            publishOdometry();
            // 发布里程计、点云、轨迹
            // 1、发布历史关键帧位姿集合
            // 2、发布局部map的降采样平面点集合
            // 3、发布历史帧（累加的）的角点、平面点降采样集合
            // 4、发布里程计轨迹
            publishFrames();
            
#if XMLLOG
            cout<<"保存xml与否,mapOptmization:"<<doc->SaveFile(logSavePath+"LM_log"+to_string(seq)+".xml")<<endl;
            doc->Clear();
#endif
            //logname.close();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {  
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    bool saveMapService(slio_sam::save_mapRequest& req, slio_sam::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        // save corner cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        slio_sam::save_mapRequest  req;
        slio_sam::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }
    void updateInitialGuess()
    {
        //ROS_INFO("updateInitialGuess  更新初始猜测");
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        ros::Time nowtime = ros::Time().now();
        if (cloudKeyPoses3D->points.empty())
        {  
            cout<<"cloudKeyPoses3D->points.empty()"<<endl;
             //对于没有imu数据的rpy来说，初始化的时候保持直走
            transformTobeMapped[0] = cloudInfo.imuRollInit;//这个还是要提供的，如果初始位姿不好 其优化结果就也不好，
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;
            //cout<<"updateInitialGuess"<<endl;
            //cout<< transformTobeMapped[0]<<" "<< transformTobeMapped[1]<<" "<< transformTobeMapped[2]<<endl;
            if (!useImuHeadingInitialization)//没有gps的时候useImuHeadingInitialization值为false
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            //ROS_INFO("updateInitialGuess 修改初始的猜测");
            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        //cout<<"cloudInfo.odomAvailable:"<<cloudInfo.odomAvailable<<" cloudInfo.imuAvailable"<<cloudInfo.imuAvailable<<endl;
        if (cloudInfo.odomAvailable == true)//这个是imuPreintegration发布出去的
        {//cloudInfo.initialGuessX这个是来源与mapOptmization的odometry_incremental
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            //cout<<"cloudInfo.odomAvailable == true: ";
            if (lastImuPreTransAvailable == false)
            {
                //cout<<"lastImuPreTransAvailable == false     ros::Time:"<<nowtime.sec<<","<<nowtime.nsec<<endl;
                lastImuPreTransformation = transBack;//lastImuPreTransformation只与transBack有关
                lastImuPreTransAvailable = true;
            } else {//astImuPreTransAvailable 存放的是上一帧的transBack，这两者都是由imuPreintegration得到的，
                //cout<<"lastImuPreTransAvailable == true     ros::Time:"<<nowtime.sec<<","<<nowtime.nsec<<endl;
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;//第一次因为lastImuPreTransformation==transBack 所以为0，后面就相当于transBack-lastImuPreTransformation
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);//对于已经运行到中段的系统来说，应该已经走到了这里。
                Eigen::Affine3f transFinal = transTobe * transIncre;//translncre 代表着由imu预积分得到的当前帧与上一帧的增量，transTobe代表着上一帧经过LM优化后的结果
                //transIncre 在运行到175-177帧的时候绝对会出问题
                //将transIncre放入Tmslncre进行记录 start
#if XMLLOG
                TiXmlElement *TmsIncre = new TiXmlElement("TmsIncre");
                frame->LinkEndChild(TmsIncre);
                float transformTobeMappedIncre[6]={0};
                pcl::getTranslationAndEulerAngles(transIncre,transformTobeMappedIncre[3],transformTobeMappedIncre[4],transformTobeMappedIncre[5],
                transformTobeMappedIncre[0],transformTobeMappedIncre[1],transformTobeMappedIncre[2]);
                TmsIncre->SetAttribute("rx",convertToString(transformTobeMappedIncre[0]));
                TmsIncre->SetAttribute("ry",convertToString(transformTobeMappedIncre[1]));
                TmsIncre->SetAttribute("rz",convertToString(transformTobeMappedIncre[2]));
                TmsIncre->SetAttribute("x",convertToString(transformTobeMappedIncre[3]));
                TmsIncre->SetAttribute("y",convertToString(transformTobeMappedIncre[4]));
                TmsIncre->SetAttribute("z",convertToString(transformTobeMappedIncre[5]));
#endif
                //将transIncre放入Tmslncre进行记录 end
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                //cout<<"odomAvailable==true， transformTobeMapped:"<< transformTobeMapped[0]<<" "<< transformTobeMapped[1]<<" "<< transformTobeMapped[2]<<" "
               // << transformTobeMapped[3]<<" "<< transformTobeMapped[4]<<"  "<< transformTobeMapped[5]<<endl;
                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // 使用imu增量估计进行姿势猜测,只是旋转
        if (cloudInfo.imuAvailable == true)
        {
            //cout<<"cloudInfo.imuAvailable == true     ros::Time:"<<nowtime.sec<<","<<nowtime.nsec<<endl;
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            //cout<<"cloudInfo.imuAvailable"<< transformTobeMapped[0]<<" "<< transformTobeMapped[1]<<" "<< transformTobeMapped[2]<<" "
            //    << transformTobeMapped[3]<<" "<< transformTobeMapped[4]<<"  "<< transformTobeMapped[5]<<endl;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        // 提取cloudKeyPoses3D 附近的点云帧, 放到 surroundingKeyPoses 中
        //ROS_INFO(" 提取cloudKeyPoses3D 附近的点云帧, 放到 surroundingKeyPoses 中");
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        // create kd-tree
        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), 
        (double)surroundingKeyframeSearchRadius, // 50m范围内的关键帧
        pointSearchInd, pointSearchSqDis);
         // 将满足要求的点云帧加到 surroundingKeyPoses 中,什么是点云关键帧
        //ROS_INFO(" 将满足要求的点云帧加到 surroundingKeyPoses 中,什么是点云关键帧");
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }
         // 对surroundingKeyPoses进降采样, 保存在 surroundingKeyPosesDS
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);//输入要降采样的点云
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);//将降采样
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }
        // 根据位姿提取出角点与面点，并进行降采样，奇怪前面不是已经提取了吗？
        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {   // 离cloudKeyPoses3D最后一个点大于50米，不进行特征提取，３D?怎么改成2D
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            // intensity里保存的是这个点的索引 作为key，角点与面点组成的pair作为value
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) //如果
            {
                // transformed cloud available　如果有转化，就吧转化提出来放进去
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available  如果没有转化，就把转化放进去，transformPointCloud是个函数
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)，降采样
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }
    void extractSurroundingKeyFrames()
    {   //没有初始化，直接返回
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        // 进行关键帧提取
        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        //ROS_INFO("面降采样前数量:%d，降低采样后数量:%d",laserCloudSurfLast->size(),laserCloudSurfLastDS->size());
    }

    void updatePointAssociateToMap()
    {
       // ROS_INFO("updatePointAssociateToMap");
        //cout<<"transformTobeMapped: "<<transformTobeMapped[0]<<" "<<transformTobeMapped[1]<<" "<<transformTobeMapped[2]<<" "<<transformTobeMapped[3]<<" "<<transformTobeMapped[4]<<" "<<transformTobeMapped[5]<<endl;
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }
    void cornerOptimization(bool testTmsFlag)
    {// 更新当前位姿与地图间位姿变换,这个是什么意思?
        //ROS_INFO("遍历点云, 构建点到直线的约束");
        updatePointAssociateToMap();
// 基于OpenMP的共享内存系统的并行编程方法, 减少for循环的执行时间
       //#pragma omp parallel for num_threads(numberOfCores)
        // 遍历点云, 构建点到直线的约束
        //ROS_INFO("被选入的特征点和参数:");
        line_list.header.frame_id = odometryFrame;
        line_list.ns="line_extraction";
        line_list.type = visualization_msgs::Marker::LINE_LIST;
        line_list.action = visualization_msgs::Marker::ADD;
        line_list.pose.orientation.w = 1.0;
        line_list.scale.x=0.01;
        line_list.color.r=1.0;
        line_list.color.a=1.0;
        line_list.points.clear();

        
        coeff_list.header.frame_id = odometryFrame;
        coeff_list.ns="coeffs";
        coeff_list.type = visualization_msgs::Marker::LINE_LIST;
        coeff_list.action = visualization_msgs::Marker::ADD;
        coeff_list.pose.orientation.w = 1.0;
        coeff_list.scale.x=0.02;
        //coeff_list.scale.y=0.02;
     
        coeff_list.color.r=0.7;
        coeff_list.color.g=0.6;
        coeff_list.color.b=0.1;
        coeff_list.color.a=0.5;
        coeff_list.points.clear();
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)//因为对于单线雷达，曲率小的是线，而不是面，所以去原本的面特征里面寻找。
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            
            // 将点云从雷达坐标系变换到map坐标系
            //ROS_INFO("出问题前");
            pointAssociateToMap(&pointOri, &pointSel);
            // 在kdtree中搜索当前点的5个邻近点
            // cout<<"pointOri:"<<pointOri.x<<" "<<pointOri.y<<" "<<pointOri.z<<" "<<pointOri.intensity<<endl;
            // cout<<"pointSel:"<<pointSel.x<<" "<<pointSel.y<<" "<<pointSel.z<<" "<<pointSel.intensity<<endl;
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            //ROS_INFO("出问题后");
            //重左到右分别是基准点，取最近点的数量，取出来的点下标，取出来的点的距离。
            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
             // 要求距离都小于1m       
            if (pointSearchSqDis[4] < 0.5) {//这应该是第5个点，这个是按距离排序的,第五个点一定是最远距离
                // 计算5个点的均值坐标，记为中心点
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                // 计算协方差
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;
                // 构建协方差矩阵
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;
                // 特征值分解
                cv::eigen(matA1, matD1, matV1);
                // 如果最大的特征值相比次大特征值，大很多，认为构成了线，角点是合格的
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // 三角形的高，也就是点到直线距离
                    float ld2 = a012 / l12;
                    // 距离越大，s越小，是个距离惩罚因子（权重），注意这里ld2是不可能大于1的
                    float s = 1 - 0.9 * fabs(ld2);
                    // 点到直线的垂线段单位向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    // 点到直线距离
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {//角点以及角点到直线法线的向量
                        // 当前激光帧角点，加入匹配集合中
                        laserCloudOriCornerVec[i] = pointOri;//因为我们待会还是要采用线特征点的方法去计算，所以这里还是放在角点这里
                        // 角点的参数，
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                        geometry_msgs::Point p;
                        p.x=x1;
                        p.y=y1;
                        p.z=z1;
                        line_list.points.push_back(p);
                        p.x=x2;
                        p.y=y2;
                        p.z=z2;
                        line_list.points.push_back(p);
                        p.x=pointSel.x;
                        p.y=pointSel.y;
                        p.z=pointSel.z;
                       // cout<<"前:"<<p<<endl;
                        coeff_list.points.push_back(p);
                        p.x-=coeff.x*coeff.intensity/(s*s);
                        p.y-=coeff.y*coeff.intensity/(s*s);
                        p.z-=coeff.z*coeff.intensity/(s*s);
                        //cout<<"后:"<<p<<endl;
                        coeff_list.points.push_back(p);
#if XMLLOG
                        if(testTmsFlag == false) {
                            TiXmlElement *point_line = new TiXmlElement("point_line");
                            iter->LinkEndChild(point_line);
                            point_line->SetAttribute("index",i);
                            point_line->SetAttribute("s",convertToString(s));
                            TiXmlElement *linexml,*coeffxml,*pointxml;
                            linexml = new TiXmlElement("line");
                            coeffxml = new TiXmlElement("coeff");
                            pointxml = new TiXmlElement("point");
                            point_line->LinkEndChild(linexml);
                            point_line->LinkEndChild(coeffxml);
                            point_line->LinkEndChild(pointxml);
                            
                            linexml->SetAttribute("x1",convertToString(x1));
                            linexml->SetAttribute("y1",convertToString(y1));
                            linexml->SetAttribute("z1",convertToString(z1));
                            linexml->SetAttribute("x2",convertToString(x2));
                            linexml->SetAttribute("y2",convertToString(y2));
                            linexml->SetAttribute("z2",convertToString(z2));
                            //logname<<"line:"<<x1<<" "<<y1<<" "<<z1<<" "<<x2<<" "<<y2<<" "<<z2<<endl;
                            coeffxml->SetAttribute("x",convertToString(coeff.x));
                            coeffxml->SetAttribute("y",convertToString(coeff.y));
                            coeffxml->SetAttribute("z",convertToString(coeff.z));
                            coeffxml->SetAttribute("intensity",convertToString(coeff.intensity));

                            pointxml->SetAttribute("x",convertToString(pointSel.x));
                            pointxml->SetAttribute("y",convertToString(pointSel.y));
                            pointxml->SetAttribute("z",convertToString(pointSel.z));
                        }
#endif
                        //logname<<"coeff:"<<coeff.x<<" "<<coeff.y<<" "<<coeff.z<<" point:"<<pointSel.x<<" "<<pointSel.y<<" "<<pointSel.z<<endl;
                        // std::cout<<"laserCorner"<<i<<":"<<pointOri.x<<" "<<pointOri.y<<" "<<pointOri.z<<"  line p1:"<<x1<<" "<<y1<<" "<<z1<<" p2:"<<x2<<" "<<y2<<" "<<z2
                        // <<" coeff:"<<coeff.x<<" "<<coeff.y<<" "<<coeff.z<<" d:"<<coeff.intensity<<endl;
                        //publishCloud(const ros::Publisher& thisPub, const T& thisCloud, ros::Time thisStamp, std::string thisFrame)
                    
                    }
                }
            }
        }
        marker_pub.publish(line_list);
        coeff_pub.publish(coeff_list);
    }
// 将2种残差添加到同一个变量中
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        //ROS_INFO("将2种残差添加到同一个变量中");
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){//因为我们用到的是线点，所以for循环用到的是surf，将我们的线点用lio的角点方式构建参差并且计算
            if (laserCloudOriCornerFlag[i] == true){//但是实际上对应lio来说线点的计算方式是corner,所以我们将线点加入corner,flag数组也在上一个函数被修改
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
            //coeffSel pub
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        //std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }
void saveStatus(){
    savelaserCloudOri = *laserCloudOri;
    savecoeffSel = *coeffSel;
    // transformTobeMapped_last[0]=transformTobeMapped[0];
    // transformTobeMapped_last[1]=transformTobeMapped[1];
    // transformTobeMapped_last[2]=transformTobeMapped[2];
    // transformTobeMapped_last[3]=transformTobeMapped[3];
    // transformTobeMapped_last[4]=transformTobeMapped[4];
    // transformTobeMapped_last[5]=transformTobeMapped[5];
}
void recoverStatus(){
    *laserCloudOri = savelaserCloudOri;
    *coeffSel = savecoeffSel;
    // transformTobeMapped[0]=transformTobeMapped_last[0];
    // transformTobeMapped[1]=transformTobeMapped_last[1];
    // transformTobeMapped[2]=transformTobeMapped_last[2];
    // transformTobeMapped[3]=transformTobeMapped_last[3];
    // transformTobeMapped[4]=transformTobeMapped_last[4];
    // transformTobeMapped[5]=transformTobeMapped_last[5];
}
double getChi(){
    int laserCloudSelNum = laserCloudOri->size();
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matBt(1,laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matBtB(1,1, CV_32F, cv::Scalar::all(0));
    PointType coeff;
    for (int i = 0; i < laserCloudSelNum; i++) {
        coeff = coeffSel->points[i];
        matB.at<float>(i, 0) = -coeff.intensity;
    }
    cv::transpose(matB, matBt);
    matBtB=matBt*matB;
    return 0.5*matBtB.at<float>(0,0);
}
double testTms(){//检测此次优化是否成功
    saveStatus();
    double last_chi = getChi();
    laserCloudOri->clear();
    coeffSel->clear();
    cornerOptimization(true);
    combineOptimizationCoeffs();
    double current_chi = getChi();
    recoverStatus();
    return last_chi - current_chi;
}
void copyiter(TiXmlElement *iter,TiXmlElement *diriter){
    for(TiXmlElement* elem = iter->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement()){
        string elem_name = elem->Value();
        //cout<<"point_line"<<endl;
        if(strcmp(elem_name.c_str(),"point_line")==0)
        {
            
            TiXmlElement *point_line = new TiXmlElement("point_line"); 
            diriter->LinkEndChild(point_line);
            point_line->SetAttribute("index",elem->Attribute("index"));
            point_line->SetAttribute("s",elem->Attribute("s"));
            for(TiXmlElement* elem2= elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement())
            {
                if(strcmp(elem2->Value(),"line")==0){
                    TiXmlElement *line = new TiXmlElement("line");
                    point_line->LinkEndChild(line);
                    geometry_msgs::Point point1,point2;
                    line->SetAttribute("x1",elem2->Attribute("x1"));
                    line->SetAttribute("y1",elem2->Attribute("y1"));
                    line->SetAttribute("z1",elem2->Attribute("z1"));
                    line->SetAttribute("x2",elem2->Attribute("x2"));
                    line->SetAttribute("y2",elem2->Attribute("y2"));
                    line->SetAttribute("z2",elem2->Attribute("z2"));       
                }
                if(strcmp(elem2->Value(),"coeff")==0){
                    TiXmlElement *coeff = new TiXmlElement("coeff");
                    point_line->LinkEndChild(coeff);
                    coeff->SetAttribute("intensity",elem2->Attribute("intensity"));
                    coeff->SetAttribute("x",elem2->Attribute("x"));
                    coeff->SetAttribute("y",elem2->Attribute("y"));
                    coeff->SetAttribute("z",elem2->Attribute("z"));
                }
                if(strcmp(elem2->Value(),"point")==0){
                    TiXmlElement *point = new TiXmlElement("point");
                    point_line->LinkEndChild(point);
                    point->SetAttribute("x",elem2->Attribute("x"));
                    point->SetAttribute("y",elem2->Attribute("y"));
                    point->SetAttribute("z",elem2->Attribute("z"));
                }
            }
        }
    }
}
int LMOptimization(int iterCount,double &currentLambda,bool inital){//这个是一点点改的
// if(iterCount == 0){
//     transformTobeMapped[2]+=0.1;//这里故意给它加了
//     transformTobeMapped[3]+=0.1;
//     transformTobeMapped[4]+=0.1; //偏差太小所以人为增加偏差，查看H矩阵   
// }

    float sx = sin(transformTobeMapped[0]);
    float cx = cos(transformTobeMapped[0]);
    float sy = sin(transformTobeMapped[1]);
    float cy = cos(transformTobeMapped[1]);
    float sz = sin(transformTobeMapped[2]);
    float cz = cos(transformTobeMapped[2]);
    #if  XMLLOG
        TiXmlElement *Tms_start,*Tms_end,*MatJ,*MatH,*MatE,*MatV,*MatP,*MatAtB,*detlaXxml;
        Tms_start = new  TiXmlElement("Tms_start");
        MatJ = new TiXmlElement("MatJ");
        MatH = new TiXmlElement("MatH");
        MatE = new TiXmlElement("MatE");
        MatV = new TiXmlElement("MatV");
        MatP = new TiXmlElement("MatP");
        MatAtB = new TiXmlElement("MatAtB");
        detlaXxml = new TiXmlElement("detlaX");
        Tms_end = new TiXmlElement("Tms_end");
        //if(iterCount==0) frame->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_end);
        iter->LinkEndChild(detlaXxml);
        iter->LinkEndChild(MatE);
        iter->LinkEndChild(MatH);
        iter->LinkEndChild(MatV);
        iter->LinkEndChild(MatP);
        iter->LinkEndChild(MatAtB);
        iter->LinkEndChild(MatJ);
        
        Tms_start->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_start->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_start->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_start->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_start->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_start->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matBt(1,laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matBtB(1,1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        //cv::Mat matAMax(1,6,CV_32F, cv::Scalar::all(0));
        PointType pointOri, coeff;
        // 遍历匹配特征点，构建Jacobian矩阵
        for (int i = 0; i < laserCloudSelNum; i++) {
            pointOri= laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            double arx = (0*pointOri.x+(cz*sy*cx+sz*sx)*pointOri.y+(sz*cx*-sx*cz*sy)*pointOri.z)*coeff.x+
            (0*pointOri.x+(-cz*sx+sz*sy*cx)*pointOri.y+(-sz*sy*sx-cz*cx)*pointOri.z)*coeff.y+
            (0*pointOri.x+(cx*cy)*pointOri.y+(-sx*cy)*pointOri.z)*coeff.z;

            double ary = ((-cz*sy)*pointOri.x+(cz*cy*sx)*pointOri.y+(cx*cz*cy)*pointOri.z)*coeff.x+
            ((-sz*sy)*pointOri.x+(sz*cy*sx)*pointOri.y+(sz*cy*cx)*pointOri.z)*coeff.y+
            ((-cy)*pointOri.x+(-sx*sy)*pointOri.y+(-cx*sy)*pointOri.z)*coeff.z;

            double arz = ((-sz*cy)*pointOri.x+(-sz*sy*sx-cz*cx)*pointOri.y+(cz*sx-cx*sz*sy)*pointOri.z)*coeff.x+
            ((cz*cy)*pointOri.x+(-sz*cx+cz*sy*sx)*pointOri.y+(cz*sy*cx+sz*sx)*pointOri.z)*coeff.y;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        int msize=0;
        // int alterind[6];
        //  for(int i=0;i<6;++i){
        //      cout<<matAMax.at<float>(0,i)<<" ";
        //      if(matAMax.at<float>(0,i)>10) {
        //          msize++;
        //      }
        //  }
         //cout<<endl;
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        
        //LM 算法 阻尼因子 start
        cv::transpose(matB, matBt);
        matBtB=matBt*matB;
        double chi = 0.5*matBtB.at<float>(0,0); 
        if(inital) {
            double maxDiagonal =-1;
            for (ulong i = 0; i < 6; ++i) {
                if(fabs(matAtA.at<float>(i, i)>maxDiagonal)) 
                maxDiagonal = fabs(matAtA.at<float>(i, i));
            }
            currentLambda = 1e-4*maxDiagonal;
        }
        matAtA+=currentLambda*cv::Mat::eye(6, 6, CV_32F);
         //LM 算法 阻尼因子 end
        // for(int i=0;i<6;++i){
        //     cout<<matAtA.at<float>(i,i)<<" ";
        // }
        // cout<<endl;
         // J^T·J·delta_x = -J^T·f 高斯牛顿
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
        //偷偷把z改成0，咱们就先不优化z了
        //matX.at<float>(0,5) = 0;
        cv::Mat matE(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
        cv::eigen(matAtA, matE, matV);//matE为特征值 matV 为特征向量
 #if XMLLOG       
        iter->SetAttribute("features_num",laserCloudSelNum);
        int rows,cols;
        rows = matA.size().height;
        for(int j = 0;j<rows;++j){
            TiXmlElement *Jxml = new TiXmlElement("J");
            MatJ->LinkEndChild(Jxml); 
            Jxml->SetAttribute("index",j);
            Jxml->SetAttribute("rx",convertToString(matA.at<float>(j,0)));
            Jxml->SetAttribute("ry",convertToString(matA.at<float>(j,1)));
            Jxml->SetAttribute("rz",convertToString(matA.at<float> (j,2)));
            Jxml->SetAttribute("x"  ,convertToString(matA.at<float>(j,3)));
            Jxml->SetAttribute("y"  ,convertToString(matA.at<float>(j,4)));
            Jxml->SetAttribute("z"  ,convertToString(matA.at<float>(j,5)));
            
        }
        rows=matAtA.size().height;
        cols  =matAtA.size().width;
        for(int j=0;j<rows;++j){
            for(int k=0;k<cols;++k){
                MatH->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matAtA.at<float>(j,k))); 
            }
        }
        
        for(int j=0;j<6;++j){
            MatE->SetAttribute("E"+to_string(j),convertToString(matE.at<float>(j,0)));
            MatAtB->SetAttribute("M"+to_string(j)+to_string(0),convertToString(matAtB.at<float>(j,0)));
        }
        rows=matV.size().height;
        cols  =matV.size().width;
        for(int j=0;j<rows;++j){
            for(int k=0;k<cols;++k){
                MatV->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matV.at<float>(j,k))); 
            }
        }
        
        
#endif
        if (true) {//修改看看会怎么样
            matV.copyTo(matV2);
            isDegenerate = false;
            float eignThre[6] = {1, 1, 1, 1, 1, 1};//为什么特征值小于100 就表示退化了， 而且是从后往前，不应该是为0表示退化米。 
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }

            matP = matV.inv() * matV2;
#if XMLLOG
            rows=matP.size().height;
            cols  =matP.size().width;
            for(int j=0;j<rows;++j){
                for(int k=0;k<cols;++k){
                    MatP->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matP.at<float>(j,k))); 
                }
            }
#endif
            //logname<<"matP:"<<matP<<endl;
        }

        if (isDegenerate)//退化
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        if(iterCount ==0 ) theta = 1;
        //matX=matAtB;



        // if(chi>1.0)
        //     matX=matAtB*0.001*chi;
        matX*=theta;//直接用梯度下降法，看看会有什么结果
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
#if XMLLOG
        detlaXxml->SetAttribute("rx",convertToString(matX.at<float>(0,0)));
        detlaXxml->SetAttribute("ry",convertToString(matX.at<float>(0,1)));
        detlaXxml->SetAttribute("rz",convertToString(matX.at<float>(0,2)));
        detlaXxml->SetAttribute("x",convertToString(matX.at<float>(0,3)));
        detlaXxml->SetAttribute("y",convertToString(matX.at<float>(0,4)));
        detlaXxml->SetAttribute("z",convertToString(matX.at<float>(0,5)));
        
        Tms_end->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_end->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_end->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_end->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_end->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_end->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
#if XMLLOG
        iter->SetAttribute("deltaR",convertToString(deltaR));
        iter->SetAttribute("deltaT",convertToString(deltaT));
#endif
        //logname<<iterCount<<"_下降量: R:"<<deltaR<<" T:"<<deltaT<<endl;
        //意思是如果修改的角度的平方小于1度^2，位移小于1cm^2

#if   TIME_TEST
time_t start=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#endif
        double deltachi=testTms();//返回的结果三last_chi - current_chi
#if   TIME_TEST
time_t end=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
cout<<"耗时"<<(long long)(end-start)<<"纳秒"<<endl;
#endif
#if XMLLOG
        iter->SetAttribute("chi",convertToString(chi));
        iter->SetAttribute("deltachi",convertToString(deltachi));
#endif
        
        if(deltachi>0){
            theta*=1.5;
            return 0;
        }
        if (deltaR < 0.01 && deltaT < 0.01) {
            //logname<<endl;
            if(deltachi<0) {
                transformTobeMapped[0] -= matX.at<float>(0, 0);
                transformTobeMapped[1] -= matX.at<float>(1, 0);
                transformTobeMapped[2] -= matX.at<float>(2, 0);
                transformTobeMapped[3] -= matX.at<float>(3, 0);
                transformTobeMapped[4] -= matX.at<float>(4, 0);
                transformTobeMapped[5] -= matX.at<float>(5, 0);
            }
#if XMLLOG
            frame->SetAttribute("last_chi",convertToString(chi));
            frame->SetAttribute("last_iterCount",iterCount);
#endif
            return 2; // converged
        }
        if(deltachi<0){//回滚
            transformTobeMapped[0] -= matX.at<float>(0, 0);
            transformTobeMapped[1] -= matX.at<float>(1, 0);
            transformTobeMapped[2] -= matX.at<float>(2, 0);
            transformTobeMapped[3] -= matX.at<float>(3, 0);
            transformTobeMapped[4] -= matX.at<float>(4, 0);
            transformTobeMapped[5] -= matX.at<float>(5, 0);
            theta*=0.5;
            return 1;
        }
        if(deltachi<1e-5){
#if XMLLOG
            frame->SetAttribute("last_chi",convertToString(chi));
            frame->SetAttribute("last_iterCount",iterCount);
# endif
            return 2;
        }


        //logname<<"chi_"<<iterCount<<": "<<0.5*matBtB.at<float>(0,0)<<endl;
        //if(iterCount==29) logname<<endl; 

        //return false; // keep optimizing
}


int LMOptimization(int iterCount){//这个是一点点改的
// if(iterCount == 0){
//     transformTobeMapped[2]+=0.1;//这里故意给它加了
//     transformTobeMapped[3]+=0.1;
//     transformTobeMapped[4]+=0.1; //偏差太小所以人为增加偏差，查看H矩阵   
// }

    float sx = sin(transformTobeMapped[0]);
    float cx = cos(transformTobeMapped[0]);
    float sy = sin(transformTobeMapped[1]);
    float cy = cos(transformTobeMapped[1]);
    float sz = sin(transformTobeMapped[2]);
    float cz = cos(transformTobeMapped[2]);
    #if  XMLLOG
        TiXmlElement *Tms_start,*Tms_end,*MatJ,*MatH,*MatE,*MatV,*MatP,*MatAtB,*detlaXxml;
        Tms_start = new  TiXmlElement("Tms_start");
        MatJ = new TiXmlElement("MatJ");
        MatH = new TiXmlElement("MatH");
        MatE = new TiXmlElement("MatE");
        MatV = new TiXmlElement("MatV");
        MatP = new TiXmlElement("MatP");
        MatAtB = new TiXmlElement("MatAtB");
        detlaXxml = new TiXmlElement("detlaX");
        Tms_end = new TiXmlElement("Tms_end");
        //if(iterCount==0) frame->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_end);
        iter->LinkEndChild(detlaXxml);
        iter->LinkEndChild(MatE);
        iter->LinkEndChild(MatH);
        iter->LinkEndChild(MatV);
        iter->LinkEndChild(MatP);
        iter->LinkEndChild(MatAtB);
        iter->LinkEndChild(MatJ);
        
        Tms_start->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_start->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_start->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_start->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_start->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_start->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matBt(1,laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matBtB(1,1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        //cv::Mat matAMax(1,6,CV_32F, cv::Scalar::all(0));
        PointType pointOri, coeff;
        // 遍历匹配特征点，构建Jacobian矩阵
        for (int i = 0; i < laserCloudSelNum; i++) {
            pointOri= laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            double arx = (0*pointOri.x+(cz*sy*cx+sz*sx)*pointOri.y+(sz*cx*-sx*cz*sy)*pointOri.z)*coeff.x+
            (0*pointOri.x+(-cz*sx+sz*sy*cx)*pointOri.y+(-sz*sy*sx-cz*cx)*pointOri.z)*coeff.y+
            (0*pointOri.x+(cx*cy)*pointOri.y+(-sx*cy)*pointOri.z)*coeff.z;

            double ary = ((-cz*sy)*pointOri.x+(cz*cy*sx)*pointOri.y+(cx*cz*cy)*pointOri.z)*coeff.x+
            ((-sz*sy)*pointOri.x+(sz*cy*sx)*pointOri.y+(sz*cy*cx)*pointOri.z)*coeff.y+
            ((-cy)*pointOri.x+(-sx*sy)*pointOri.y+(-cx*sy)*pointOri.z)*coeff.z;

            double arz = ((-sz*cy)*pointOri.x+(-sz*sy*sx-cz*cx)*pointOri.y+(cz*sx-cx*sz*sy)*pointOri.z)*coeff.x+
            ((cz*cy)*pointOri.x+(-sz*cx+cz*sy*sx)*pointOri.y+(cz*sy*cx+sz*sx)*pointOri.z)*coeff.y;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        int msize=0;
        // int alterind[6];
        //  for(int i=0;i<6;++i){
        //      cout<<matAMax.at<float>(0,i)<<" ";
        //      if(matAMax.at<float>(0,i)>10) {
        //          msize++;
        //      }
        //  }
         //cout<<endl;
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        
        //LM 算法 阻尼因子

        // for(int i=0;i<6;++i){
        //     cout<<matAtA.at<float>(i,i)<<" ";
        // }
        // cout<<endl;
         // J^T·J·delta_x = -J^T·f 高斯牛顿
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
        //偷偷把z改成0，咱们就先不优化z了
        //matX.at<float>(0,5) = 0;
        cv::Mat matE(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
        cv::eigen(matAtA, matE, matV);//matE为特征值 matV 为特征向量
 #if XMLLOG       
        iter->SetAttribute("features_num",laserCloudSelNum);
        int rows,cols;
        rows = matA.size().height;
        for(int j = 0;j<rows;++j){
            TiXmlElement *Jxml = new TiXmlElement("J");
            MatJ->LinkEndChild(Jxml); 
            Jxml->SetAttribute("index",j);
            Jxml->SetAttribute("rx",convertToString(matA.at<float>(j,0)));
            Jxml->SetAttribute("ry",convertToString(matA.at<float>(j,1)));
            Jxml->SetAttribute("rz",convertToString(matA.at<float> (j,2)));
            Jxml->SetAttribute("x"  ,convertToString(matA.at<float>(j,3)));
            Jxml->SetAttribute("y"  ,convertToString(matA.at<float>(j,4)));
            Jxml->SetAttribute("z"  ,convertToString(matA.at<float>(j,5)));
            
        }
        rows=matAtA.size().height;
        cols  =matAtA.size().width;
        for(int j=0;j<rows;++j){
            for(int k=0;k<cols;++k){
                MatH->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matAtA.at<float>(j,k))); 
            }
        }
        
        for(int j=0;j<6;++j){
            MatE->SetAttribute("E"+to_string(j),convertToString(matE.at<float>(j,0)));
            MatAtB->SetAttribute("M"+to_string(j)+to_string(0),convertToString(matAtB.at<float>(j,0)));
        }
        rows=matV.size().height;
        cols  =matV.size().width;
        for(int j=0;j<rows;++j){
            for(int k=0;k<cols;++k){
                MatV->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matV.at<float>(j,k))); 
            }
        }
        
        
#endif
        if (true) {//修改看看会怎么样
            matV.copyTo(matV2);
            isDegenerate = false;
            float eignThre[6] = {1, 1, 1, 1, 1, 1};//为什么特征值小于100 就表示退化了， 而且是从后往前，不应该是为0表示退化米。 
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }

            matP = matV.inv() * matV2;
#if XMLLOG
            rows=matP.size().height;
            cols  =matP.size().width;
            for(int j=0;j<rows;++j){
                for(int k=0;k<cols;++k){
                    MatP->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matP.at<float>(j,k))); 
                }
            }
#endif
            //logname<<"matP:"<<matP<<endl;
        }

        if (isDegenerate)//退化
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        if(iterCount ==0 ) theta = 1;
        //matX=matAtB;
        cv::transpose(matB, matBt);
        matBtB=matBt*matB;
        double chi = 0.5*matBtB.at<float>(0,0); 
        matX*=theta;//直接用梯度下降法，看看会有什么结果
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
#if XMLLOG
        detlaXxml->SetAttribute("rx",convertToString(matX.at<float>(0,0)));
        detlaXxml->SetAttribute("ry",convertToString(matX.at<float>(0,1)));
        detlaXxml->SetAttribute("rz",convertToString(matX.at<float>(0,2)));
        detlaXxml->SetAttribute("x",convertToString(matX.at<float>(0,3)));
        detlaXxml->SetAttribute("y",convertToString(matX.at<float>(0,4)));
        detlaXxml->SetAttribute("z",convertToString(matX.at<float>(0,5)));
        
        Tms_end->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_end->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_end->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_end->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_end->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_end->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
#if XMLLOG
        iter->SetAttribute("deltaR",convertToString(deltaR));
        iter->SetAttribute("deltaT",convertToString(deltaT));
#endif
        //logname<<iterCount<<"_下降量: R:"<<deltaR<<" T:"<<deltaT<<endl;
        //意思是如果修改的角度的平方小于1度^2，位移小于1cm^2

#if   TIME_TEST
time_t start=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#endif
        double deltachi=testTms();//返回的结果三last_chi - current_chi
#if   TIME_TEST
time_t end=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
cout<<"耗时"<<(long long)(end-start)<<"纳秒"<<endl;
#endif
#if XMLLOG
        iter->SetAttribute("chi",convertToString(chi));
        iter->SetAttribute("deltachi",convertToString(deltachi));
#endif
        
        if(deltachi>0){
            theta*=1.5;
            return 0;
        }
        if (deltaR < 0.01 && deltaT < 0.01) {
            //logname<<endl;
            if(deltachi<0) {
                transformTobeMapped[0] -= matX.at<float>(0, 0);
                transformTobeMapped[1] -= matX.at<float>(1, 0);
                transformTobeMapped[2] -= matX.at<float>(2, 0);
                transformTobeMapped[3] -= matX.at<float>(3, 0);
                transformTobeMapped[4] -= matX.at<float>(4, 0);
                transformTobeMapped[5] -= matX.at<float>(5, 0);
            }
#if XMLLOG
            frame->SetAttribute("last_chi",convertToString(chi));
            frame->SetAttribute("last_iterCount",iterCount);
#endif
            return 2; // converged
        }
        if(deltachi<0){//回滚
            transformTobeMapped[0] -= matX.at<float>(0, 0);
            transformTobeMapped[1] -= matX.at<float>(1, 0);
            transformTobeMapped[2] -= matX.at<float>(2, 0);
            transformTobeMapped[3] -= matX.at<float>(3, 0);
            transformTobeMapped[4] -= matX.at<float>(4, 0);
            transformTobeMapped[5] -= matX.at<float>(5, 0);
            theta*=0.5;
            return 1;
        }
        if(deltachi<1e-5){
#if XMLLOG
            frame->SetAttribute("last_chi",convertToString(chi));
            frame->SetAttribute("last_iterCount",iterCount);
# endif
            return 2;
        }


        //logname<<"chi_"<<iterCount<<": "<<0.5*matBtB.at<float>(0,0)<<endl;
        //if(iterCount==29) logname<<endl; 

        //return false; // keep optimizing
}
bool LMOptimization3(int iterCount)// 这个是参考，不能用的
{
        cout<<"LMOptimization1"<<endl;
        float sx = sin(transformTobeMapped[0]);
        float cx = cos(transformTobeMapped[0]);
        float sy = sin(transformTobeMapped[1]);
        float cy = cos(transformTobeMapped[1]);
        float sz = sin(transformTobeMapped[2]);
        float cz = cos(transformTobeMapped[2]);
#if  XMLLOG
        TiXmlElement *Tms_start,*Tms_end,*MatJ,*MatH,*MatE,*MatV,*MatP,*detlaXxml;
        Tms_start = new  TiXmlElement("Tms_start");
        MatJ = new TiXmlElement("MatJ");
        MatH = new TiXmlElement("MatH");
        MatE = new TiXmlElement("MatE");
        MatV = new TiXmlElement("MatV");
        MatP = new TiXmlElement("MatP");
        detlaXxml = new TiXmlElement("detlaX");
        Tms_end = new TiXmlElement("Tms_end");
        if(iterCount==0) frame->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_end);
        iter->LinkEndChild(detlaXxml);
        iter->LinkEndChild(MatE);
        iter->LinkEndChild(MatH);
        iter->LinkEndChild(MatV);
        iter->LinkEndChild(MatP);
        iter->LinkEndChild(MatJ);
        
        
        
        
        Tms_start->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_start->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_start->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_start->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_start->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_start->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif

        /*
        logname<<"Tms_start:"<<endl<<transformTobeMapped[0]<<" "
    <<transformTobeMapped[1]<<" "
    <<transformTobeMapped[2]<<" "
    <<transformTobeMapped[3]<<" "
    <<transformTobeMapped[4]<<" "
    <<transformTobeMapped[5]<<endl;
    */
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }
        
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        //cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matBt(1,laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matBtB(1,1, CV_32F, cv::Scalar::all(0));
        //cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAMax(1,6,CV_32F, cv::Scalar::all(0));
        PointType pointOri, coeff;
        // 遍历匹配特征点，构建Jacobian矩阵

        cout<<"LMOptimization2"<<endl;
        for (int i = 0; i < laserCloudSelNum; i++) {
            //time_t start=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            pointOri= laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            double arx = (0*pointOri.x+(cx*sy*cx)*pointOri.y+(sx*cx*-sx*cz*cy)*pointOri.z)*coeff.x+
            (0*pointOri.x+(-cz*sx+sz*sy*cx)*pointOri.y+(-sz*sy*sx-cz*cx)*pointOri.z)*coeff.y+
            (0*pointOri.x+(cx*cy)*pointOri.y+(-sx*cy)*pointOri.z)*coeff.z;

            double ary = ((-cz*sy)*pointOri.x+(cz*cy*sx)*pointOri.y+(cx*cz*cy)*pointOri.z)*coeff.x+
            ((-sz*sy)*pointOri.x+(sz*cy*sx)*pointOri.y+(sz*cy*cx)*pointOri.z)*coeff.y+
            ((-cy)*pointOri.x+(-sx*sy)*pointOri.y+(-cx*sy)*pointOri.z)*coeff.z;

            double arz = ((-sz*cy)*pointOri.x+(-sz*sy*sx-cz)*pointOri.y+(cz*sx-cx*sz*sy)*pointOri.z)*coeff.x+
            ((cx*cy)*pointOri.x+(-sz*cx+cz*sy*sx)*pointOri.y+(cz*sy*cx+sz*sx)*pointOri.z)*coeff.y;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
            matAMax.at<float>(0,0)+=arx*arx;
            matAMax.at<float>(0,1)+=ary*ary;
            matAMax.at<float>(0,2)+=arz*arz;
            matAMax.at<float>(0,3)+=coeff.x*coeff.x;
            matAMax.at<float>(0,4)+=coeff.y*coeff.y;
            matAMax.at<float>(0,5)+=coeff.z*coeff.z;
            //time_t end=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            //cout<<"耗时"<<(long long)(end-start)<<"纳秒"<<endl;
        }
        cout<<"LMOptimization3"<<endl;
        int msize=0;
        // int alterind[6];
         for(int i=0;i<6;++i){
             cout<<matAMax.at<float>(0,i)<<" ";
             if(matAMax.at<float>(0,i)>10) {
                 msize++;
             }
         }
         cout<<endl;
         cout<<"LMOptimization3.1"<<endl;
        cv::Mat matAr(laserCloudSelNum, msize, CV_32F, cv::Scalar::all(0));//表示去除变化量小的列，意思是不对几乎没有变化的状态进行优化
        vector<int> mindex(msize);//恢复矩阵的时候需要用到
        for(int i=0,k=0;i<6;++i){
            if(matAMax.at<float>(0,i)>10){
                for(int j=0;j<laserCloudSelNum;++j){
                    matAr.at<float>(j,k)=matA.at<float>(j,i);
                }
                mindex[k++]=i;
            }
        }
        cout<<"LMOptimization3.2"<<endl;
        cv::Mat matArt(msize,laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cout<<"LMOptimization3.3"<<endl;
        cv::transpose(matAr, matArt);
        cv::Mat matArtAr(msize,msize,CV_32F,cv::Scalar::all(0));
        cv::Mat matArtB(msize,1,CV_32F,cv::Scalar::all(0));
        cv::Mat matXr(msize,1,CV_32F,cv::Scalar::all(0));
        cout<<"LMOptimization3.4"<<endl;
        matArtAr = matAt * matA;
        matArtB = matAt * matB;
         // J^T·J·delta_x = -J^T·f 高斯牛顿
        cout<<"LMOptimization3.5"<<endl;
         
        cv::solve(matArtAr, matArtB, matXr, cv::DECOMP_QR);
        for(int i=0;i<msize;++i){
            matX.at<float>(mindex[i],0)=matXr.at<float>(i,0);
        }
        cout<<"LMOptimization4"<<endl;
 #if XMLLOG       
        iter->SetAttribute("features_num",laserCloudSelNum);
        int rows,cols;
        rows = matA.size().height;
        for(int j = 0;j<rows;++j){
            TiXmlElement *Jxml = new TiXmlElement("J");
            MatJ->LinkEndChild(Jxml); 
            Jxml->SetAttribute("index",j);
            Jxml->SetAttribute("rx",convertToString(matA.at<float>(j,0)));
            Jxml->SetAttribute("ry",convertToString(matA.at<float>(j,1)));
            Jxml->SetAttribute("rz",convertToString(matA.at<float> (j,2)));
            Jxml->SetAttribute("x"  ,convertToString(matA.at<float>(j,3)));
            Jxml->SetAttribute("y"  ,convertToString(matA.at<float>(j,4)));
            Jxml->SetAttribute("z"  ,convertToString(matA.at<float>(j,5)));
        }
        rows=matArtAr.size().height;
        cols  =matArtAr.size().width;
        for(int j=0;j<rows;++j){
            for(int k=0;k<cols;++k){
                MatH->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matArtAr.at<float>(j,k))); 
            }
        }
        // for(int j=0;j<6;++j){
        //     MatE->SetAttribute("E"+to_string(j),convertToString(matE.at<float>(j,0)));
        // }
        // rows=matV.size().height;
        // cols  =matV.size().width;
        // for(int j=0;j<rows;++j){
        //     for(int k=0;k<cols;++k){
        //         MatV->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matV.at<float>(j,k))); 
        //     }
        // }
        
        detlaXxml->SetAttribute("rx",convertToString(matX.at<float>(0,0)));
        detlaXxml->SetAttribute("ry",convertToString(matX.at<float>(1,0)));
        detlaXxml->SetAttribute("rz",convertToString(matX.at<float>(2,0)));
        detlaXxml->SetAttribute("x",convertToString(matX.at<float>(3,0)));
        detlaXxml->SetAttribute("y",convertToString(matX.at<float>(4,0)));
        detlaXxml->SetAttribute("z",convertToString(matX.at<float>(5,0)));
#endif
        cout<<"LMOptimization5"<<endl;
        /*
        logname<<"特点数量:"<<laserCloudSelNum<<"\nJ矩阵:"<<matA<<"\nH矩阵:"<<matAtA<<"\nmatE:"<<matE<<"\nmatV:"<<matV<<"\ndetalX:"<<matX<<"\n Tms_end:"<<transformTobeMapped[0]
        <<transformTobeMapped[1]<<" "
        <<transformTobeMapped[2]<<" "
        <<transformTobeMapped[3]<<" "
        <<transformTobeMapped[4]<<" "
        <<transformTobeMapped[5]<<endl;
        */



        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
#if XMLLOG
        Tms_end->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_end->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_end->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_end->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_end->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_end->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
        cout<<"LMOptimization6"<<endl;
#if XMLLOG
        iter->SetAttribute("deltaR",convertToString(deltaR));
        iter->SetAttribute("deltaT",convertToString(deltaT));
#endif
        //logname<<iterCount<<"_下降量: R:"<<deltaR<<" T:"<<deltaT<<endl;
        //意思是如果修改的角度的平方小于1度^2，位移小于1cm^2
        cv::transpose(matB, matBt);
        matBtB=matBt*matB;
#if XMLLOG
        iter->SetAttribute("chi",convertToString(0.5*matBtB.at<float>(0,0)));
#endif
        cout<<"LMOptimization7"<<endl;
        if (deltaR < 0.1 && deltaT < 0.1) {
            //logname<<endl;
            return true; // converged
        }

        //logname<<"chi_"<<iterCount<<": "<<0.5*matBtB.at<float>(0,0)<<endl;
        //if(iterCount==29) logname<<endl; 

        return false; // keep optimizing
}

bool LMOptimization2(int iterCount)//这个是标准用的，不能变,备份用
    {
        float sx = sin(transformTobeMapped[0]);
        float cx = cos(transformTobeMapped[0]);
        float sy = sin(transformTobeMapped[1]);
        float cy = cos(transformTobeMapped[1]);
        float sz = sin(transformTobeMapped[2]);
        float cz = cos(transformTobeMapped[2]);
#if  XMLLOG
        TiXmlElement *Tms_start,*Tms_end,*MatJ,*MatH,*MatE,*MatV,*MatP,*detlaXxml;
        Tms_start = new  TiXmlElement("Tms_start");
        MatJ = new TiXmlElement("MatJ");
        MatH = new TiXmlElement("MatH");
        MatE = new TiXmlElement("MatE");
        MatV = new TiXmlElement("MatV");
        MatP = new TiXmlElement("MatP");
        detlaXxml = new TiXmlElement("detlaX");
        Tms_end = new TiXmlElement("Tms_end");
        //if(iterCount==0) frame->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_start);
        iter->LinkEndChild(Tms_end);
        iter->LinkEndChild(detlaXxml);
        iter->LinkEndChild(MatE);
        iter->LinkEndChild(MatH);
        iter->LinkEndChild(MatV);
        iter->LinkEndChild(MatP);
        iter->LinkEndChild(MatJ);
        
        
        
        
        Tms_start->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_start->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_start->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_start->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_start->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_start->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
        /*
        logname<<"Tms_start:"<<endl<<transformTobeMapped[0]<<" "
    <<transformTobeMapped[1]<<" "
    <<transformTobeMapped[2]<<" "
    <<transformTobeMapped[3]<<" "
    <<transformTobeMapped[4]<<" "
    <<transformTobeMapped[5]<<endl;
    */
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }
        
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matBt(1,laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matBtB(1,1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;
        // 遍历匹配特征点，构建Jacobian矩阵

        
        for (int i = 0; i < laserCloudSelNum; i++) {
            //time_t start=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            pointOri= laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            double arx = (0*pointOri.x+(cz*sy*cx+sz*sx)*pointOri.y+(sz*cx*-sx*cz*sy)*pointOri.z)*coeff.x+
            (0*pointOri.x+(-cz*sx+sz*sy*cx)*pointOri.y+(-sz*sy*sx-cz*cx)*pointOri.z)*coeff.y+
            (0*pointOri.x+(cx*cy)*pointOri.y+(-sx*cy)*pointOri.z)*coeff.z;

            double ary = ((-cz*sy)*pointOri.x+(cz*cy*sx)*pointOri.y+(cx*cz*cy)*pointOri.z)*coeff.x+
            ((-sz*sy)*pointOri.x+(sz*cy*sx)*pointOri.y+(sz*cy*cx)*pointOri.z)*coeff.y+
            ((-cy)*pointOri.x+(-sx*sy)*pointOri.y+(-cx*sy)*pointOri.z)*coeff.z;

            double arz = ((-sz*cy)*pointOri.x+(-sz*sy*sx-cz*cx)*pointOri.y+(cz*sx-cx*sz*sy)*pointOri.z)*coeff.x+
            ((cz*cy)*pointOri.x+(-sz*cx+cz*sy*sx)*pointOri.y+(cz*sy*cx+sz*sx)*pointOri.z)*coeff.y;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
            //time_t end=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            //cout<<"耗时"<<(long long)(end-start)<<"纳秒"<<endl;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
         // J^T·J·delta_x = -J^T·f 高斯牛顿
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
        //偷偷把z改成0，咱们就先不优化z了
        //matX.at<float>(0,5) = 0;
        cv::Mat matE(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
        cv::eigen(matAtA, matE, matV);//matE为特征值 matV 为特征向量
 #if XMLLOG       
        iter->SetAttribute("features_num",laserCloudSelNum);
        int rows,cols;
        rows = matA.size().height;
        for(int j = 0;j<rows;++j){
            TiXmlElement *Jxml = new TiXmlElement("J");
            MatJ->LinkEndChild(Jxml); 
            Jxml->SetAttribute("index",j);
            Jxml->SetAttribute("rx",convertToString(matA.at<float>(j,0)));
            Jxml->SetAttribute("ry",convertToString(matA.at<float>(j,1)));
            Jxml->SetAttribute("rz",convertToString(matA.at<float> (j,2)));
            Jxml->SetAttribute("x"  ,convertToString(matA.at<float>(j,3)));
            Jxml->SetAttribute("y"  ,convertToString(matA.at<float>(j,4)));
            Jxml->SetAttribute("z"  ,convertToString(matA.at<float>(j,5)));
        }
        rows=matAtA.size().height;
        cols  =matAtA.size().width;
        for(int j=0;j<rows;++j){
            for(int k=0;k<cols;++k){
                MatH->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matAtA.at<float>(j,k))); 
            }
        }
        for(int j=0;j<6;++j){
            MatE->SetAttribute("E"+to_string(j),convertToString(matE.at<float>(j,0)));
        }
        rows=matV.size().height;
        cols  =matV.size().width;
        for(int j=0;j<rows;++j){
            for(int k=0;k<cols;++k){
                MatV->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matV.at<float>(j,k))); 
            }
        }
        
        detlaXxml->SetAttribute("rx",convertToString(matX.at<float>(0,0)));
        detlaXxml->SetAttribute("ry",convertToString(matX.at<float>(0,1)));
        detlaXxml->SetAttribute("rz",convertToString(matX.at<float>(0,2)));
        detlaXxml->SetAttribute("x",convertToString(matX.at<float>(0,3)));
        detlaXxml->SetAttribute("y",convertToString(matX.at<float>(0,4)));
        detlaXxml->SetAttribute("z",convertToString(matX.at<float>(0,5)));
#endif
        
        /*
        logname<<"特点数量:"<<laserCloudSelNum<<"\nJ矩阵:"<<matA<<"\nH矩阵:"<<matAtA<<"\nmatE:"<<matE<<"\nmatV:"<<matV<<"\ndetalX:"<<matX<<"\n Tms_end:"<<transformTobeMapped[0]
        <<transformTobeMapped[1]<<" "
        <<transformTobeMapped[2]<<" "
        <<transformTobeMapped[3]<<" "
        <<transformTobeMapped[4]<<" "
        <<transformTobeMapped[5]<<endl;
        */
        if (true) {

            // cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            // cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            // cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // cv::eigen(matAtA, matE, matV);//matE为特征值 matV 为特征向量
            matV.copyTo(matV2);
            
            isDegenerate = false;
            float eignThre[6] = {1, 1, 1, 1, 1, 1};//为什么特征值小于100 就表示退化了， 而且是从后往前，不应该是为0表示退化米。 
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
#if XMLLOG
            matP = matV.inv() * matV2;
            rows=matP.size().height;
            cols  =matP.size().width;
            for(int j=0;j<rows;++j){
                for(int k=0;k<cols;++k){
                    MatP->SetAttribute("M"+to_string(j)+to_string(k),convertToString(matP.at<float>(j,k))); 
                }
            }
#endif
            //logname<<"matP:"<<matP<<endl;
        }

        if (isDegenerate)//退化
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
#if XMLLOG
        Tms_end->SetAttribute("rx",convertToString(transformTobeMapped[0]));
        Tms_end->SetAttribute("ry",convertToString(transformTobeMapped[1]));
        Tms_end->SetAttribute("rz",convertToString(transformTobeMapped[2]));
        Tms_end->SetAttribute("x",convertToString(transformTobeMapped[3]));
        Tms_end->SetAttribute("y",convertToString(transformTobeMapped[4]));
        Tms_end->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
#if XMLLOG
        iter->SetAttribute("deltaR",convertToString(deltaR));
        iter->SetAttribute("deltaT",convertToString(deltaT));
#endif
        //logname<<iterCount<<"_下降量: R:"<<deltaR<<" T:"<<deltaT<<endl;
        //意思是如果修改的角度的平方小于1度^2，位移小于1cm^2
        cv::transpose(matB, matBt);
        matBtB=matBt*matB;
#if XMLLOG
        iter->SetAttribute("chi",convertToString(0.5*matBtB.at<float>(0,0)));
         iter->SetAttribute("deltachi",0);
#endif
        if (deltaR < 0.1 && deltaT < 0.1) {
            //logname<<endl;
            return true; // converged
        }

        //logname<<"chi_"<<iterCount<<": "<<0.5*matBtB.at<float>(0,0)<<endl;
        //if(iterCount==29) logname<<endl; 

        return false; // keep optimizing
    }
    void scan2MapOptimization()
    {// 构建点到平面、点到直线的残差, 用高斯牛顿法进行优化
        //ROS_INFO("---------------------scan2MapOptimization_start---------------");
        if (cloudKeyPoses3D->points.empty()){
           // ROS_INFO("------------cloudKeyPoses3D->points.empty()_return--------------- ");
            return;
        }
        // 降采样后的角点与面点的数量要同时大于阈值才进行优化
        // surfFeatureMinValidNum 100
        if (laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {//接下来我们要做的就是，将点到线距离改成点到点距离，点到面距离，改成点到线距离。
        //laserCloudSurfFromMap存放的是整个地图降采用的结果
            kdtreeCornerFromMap->setInputCloud(laserCloudSurfFromMapDS);
             //transformTobeMapped[2]+=0.1;//这里故意给它加了
             //transformTobeMapped[3]+=0.2;
             //transformTobeMapped[4]+=0.1; //偏差太小所以人为增加偏差，查看H矩阵   
#if XMLLOG
            TiXmlElement *Mapxml = new TiXmlElement("Map");
            TiXmlElement *Scanxml = new TiXmlElement("Scan");
            frame->LinkEndChild(Mapxml);
            frame->LinkEndChild(Scanxml);
            int indexx = 0;
            for(auto mpoint:*laserCloudSurfFromMapDS){
                TiXmlElement *point = new TiXmlElement("point");
                Mapxml->LinkEndChild(point);
                point->SetAttribute("index",indexx++);
                point->SetAttribute("x", convertToString(mpoint.x));
                point->SetAttribute("y",convertToString(mpoint.y));
                point->SetAttribute("z",convertToString(mpoint.z));
                point->SetAttribute("intensity",convertToString(mpoint.intensity));
            }
            // 进行30次迭代
            indexx=0;
            for(auto spoint:*laserCloudSurfLastDS){
                TiXmlElement *point = new TiXmlElement("point");
                Scanxml->LinkEndChild(point);
                point->SetAttribute("index",indexx++);
                point->SetAttribute("x", convertToString(spoint.x));
                point->SetAttribute("y",convertToString(spoint.y));
                point->SetAttribute("z",convertToString(spoint.z));
                point->SetAttribute("intensity",convertToString(spoint.intensity));
            }
#endif
            int LMflag = 0;
            TiXmlElement *iter_last ; 
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
#if XMLLOG
                // 点到平面, 点到直线的残差, 这里写法还与aloam有点区别
                iter = new TiXmlElement("iter");
                frame->LinkEndChild(iter);
                iter->SetAttribute("itercount",iterCount);
#endif
#if LM1
                laserCloudOri->clear();
                coeffSel->clear();
                cornerOptimization(false);
                combineOptimizationCoeffs();
                if(LMOptimization2(iterCount)){
                    break;  
                }
#endif
    #if LM2
                if(LMflag == 0){//如果迭代使结果下降,重新提取特征进行优化
                    laserCloudOri->clear();
                    coeffSel->clear();
                    cornerOptimization(false);
                    combineOptimizationCoeffs();
                    iter_last = new TiXmlElement("iter");
                #if XMLLOG
                    copyiter(iter,iter_last);//iter->iter_last
                #endif
                    LMflag=LMOptimization(iterCount);
                }else if(LMflag == 1){//如果迭代导致结果误差变大,回滚，且不重新提取特征

                #if XMLLOG
                    copyiter(iter_last,iter);//iter_last->iter
                #endif
                    LMflag=LMOptimization(iterCount);
                }
                if (LMflag == 2)//如果迭代成功
                    break;            
    #endif  
            }
            
            // 更新　transformTobeMapped
             // 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
             //看看不用imu对它有没有影响
            transformUpdate();
#if XMLLOG
            TiXmlElement *Tms_fusion = new TiXmlElement("Tms_fusion");
            frame->LinkEndChild(Tms_fusion);
            Tms_fusion->SetAttribute("rx",convertToString(transformTobeMapped[0]));
            Tms_fusion->SetAttribute("ry",convertToString(transformTobeMapped[1]));
            Tms_fusion->SetAttribute("rz",convertToString(transformTobeMapped[2]));
            Tms_fusion->SetAttribute("x",convertToString(transformTobeMapped[3]));
            Tms_fusion->SetAttribute("y",convertToString(transformTobeMapped[4]));
            Tms_fusion->SetAttribute("z",convertToString(transformTobeMapped[5]));
#endif
            //ROS_INFO("---------------------scan2MapOptimization_end---------------");
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
        
    }
    //用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {// 俯仰角小于1.4
            if (std::abs(cloudInfo.imuPitchInit) < 0.5)
            {   
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;
                //球面线性插值
                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;
                // pitch角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                // slerp pitch
                // 更新当前帧位姿的roll, pitch, z坐标；因为是小车，roll、pitch是相对稳定的，不会有很大变动，一定程度上可以信赖imu的数据，z是进行高度约束
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        // 当前帧位姿
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }
/**
 * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
*/
    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())//这个不管了,对scan还是非scan影响不大
        {// 第一帧进来时初始化gtsam参数
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 先验因子
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 二元因子
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }
/**
 * 设置当前帧为关键帧并执行因子图优化
 * 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
 * 2、添加激光里程计因子、GPS因子、闭环因子
 * 3、执行因子图优化
 * 4、得到当前帧优化后位姿，位姿协方差
 * 5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
*/
    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)//计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
            return;

        // 激光里程计因子
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
         // 优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        // 当前帧位姿结果
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        //ROS_INFO("将thisPose3D加入cloudKeyPoses3D");
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        //ROS_INFO("到了发布雷达里程计的地方publish(laserOdometryROS)");
        //位姿记录 ，用于 evo ，用tum格式记录
        ofstream foutC(pathSaveName, ios::app);//这里修改了
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(5);
        foutC << laserOdometryROS.header.stamp.toSec()<< " ";
        foutC.precision(5);
        foutC << laserOdometryROS.pose.pose.position.x << " "
              << laserOdometryROS.pose.pose.position.y << " "
              << laserOdometryROS.pose.pose.position.z << " "
              << laserOdometryROS.pose.pose.orientation.x << " "
              << laserOdometryROS.pose.pose.orientation.y << " "
              << laserOdometryROS.pose.pose.orientation.z << " "
              << laserOdometryROS.pose.pose.orientation.w <<endl;
        foutC.close();
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "base_footprint");
        //ROS_INFO("发布里程计到雷达的转换 odom to lidar_link");//现在不知道这个lidar_link有什么用
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        //ROS_INFO("发布里程计里程计增量 pubLaserOdometryIncremental");
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        //ROS_INFO("publishFrames()_start");
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                slio_sam::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudCornerLastDS;
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudCornerFromMapDS;
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
        //ROS_INFO("publishFrames()_end");
    }
};


int main(int argc, char** argv)
{
    setlocale(LC_CTYPE,"zh_CN.utf8");
    ros::init(argc, argv, "slio_sam");

    mapOptimization MO;

    //ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    ROS_INFO("\033[1;32m---->启动图优化.\033[0m");
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
