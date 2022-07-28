#include "../include/utility.h"
#include "slio_sam/cloud_info.h"
#include "sensor_msgs/LaserScan.h"
#include "tinyxml.h"
#define XMLLOG 0
#define PROCESS 0
#define PROCESS_IMU  1
/**
 * @brief ImageProjection模块的主要功能为：
    利用当前激光帧起止时刻间的imu数据计算旋转增量，
    IMU里程计数据（来自ImuPreintegration）计算平移增量，
    进而对该帧激光每一时刻的激光点进行运动畸变校正
    （利用相对于激光帧起始时刻的位姿增量，变换当前激光点到起始时刻激光点的坐标系下，实现校正）；
    同时用IMU数据的姿态角（RPY，roll、pitch、yaw）、IMU里程计数据的的位姿，对当前帧激光位姿进行粗略初始化。
 */
/**
 * Velodyne点云结构，变量名XYZIRT是每个变量的首字母
*/
struct VelodynePointXYZIT
{
    PCL_ADD_POINT4D  // 位置
    PCL_ADD_INTENSITY;  // 激光点反射强度，也可以存点的索引
    float time;  // 时间戳，记录相对于当前帧第一个激光点的时差，第一个点time=0
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;// 内存16字节对齐，EIGEN SSE优化要求
// 注册为PCL点云格式
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity) (float, time, time)
)
// 本程序使用Velodyne点云结构
using PointXYZIT = VelodynePointXYZIT;
// imu数据队列长度
const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
    // imu队列、odom队列互斥锁
    std::mutex imuLock;
    std::mutex odoLock;
    // 订阅原始激光点云
    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    // 发布当前帧校正后点云，有效点
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;
    //ros::Publisher pubLaserExtract;
    // imu数据队列（原始数据，转lidar系下）
    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;
    // imu里程计队列
    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;
    // 激光scan数据队列
    std::deque<sensor_msgs::LaserScan> scanQueue;
    // 队列front帧，作为当前处理scan帧
    sensor_msgs::LaserScan currentScanMsg;
    // 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时时间戳；
    //用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];
#if XMLLOG
    TiXmlDocument *doc;//存放日志文件，用xml 格式
    TiXmlElement *frame;
    TiXmlElement *stamp;
    TiXmlElement *imuOdomXml;
#endif 
    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;
    // 当前帧原始激光点云
    pcl::PointCloud<PointXYZIT>::Ptr laserCloudIn; //这里要改主要是类型需要修改，不过不要怕，因为只有这个文件用到了，影响不大

    // 当前帧运动畸变校正之后的激光点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    // 从fullCloud中提取有效点
    pcl::PointCloud<PointType>::Ptr   extractedCloud;
    ros::Time timeLaserInfoStamp;
//去畸变后time就没有用了
    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    // 当前激光帧起止时刻对应imu里程计位姿变换，该变换对应的平移增量；
    //  用于插值计算当前激光帧起止时间范围内，每一时刻的位置
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;
    // 当前帧激光点云运动畸变校正之后的数据，包括点云数据、初始位姿、
    //  姿态角等，发布给featureExtraction进行特征提取
    slio_sam::cloud_info cloudInfo;
    // 当前帧起始时刻
    double timeScanCur;
    // 当前帧结束时刻
    double timeScanEnd;
    // 当前帧header，包含时间戳信息
    std_msgs::Header scanHeader;

    vector<int> columnIdnCountVec;


public:
    ImageProjection():deskewFlag(0)
    {
        // 订阅原始imu数据
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅原始lidar数据
        subLaserCloud = nh.subscribe<sensor_msgs::LaserScan>(scanTopic, 5, &ImageProjection::scanHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布当前激光帧运动畸变校正后的点云，有效点
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("slio_sam/deskew/scan_deskewed", 1);
        // 发布当前激光帧运动畸变校正后的点云信息
        pubLaserCloudInfo = nh.advertise<slio_sam::cloud_info> ("slio_sam/deskew/cloud_info", 1);
        //pubLaserExtract  = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/Extract/cloud_line", 1);
         // 初始化，分配内存
        allocateMemory();
        // 重置参数
        resetParameters();
        // pcl日志级别，只打ERROR日志
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()//分配空间
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
    //assign()这里第一个参数是元素个数，第二个参数的元素值，意思是分配一个大小为N_SCAN的空间，值为0
        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }
    string convertToString(double d) {
	ostringstream os;
	    if (os << d)
		    return os.str();
	    return "invalid conversion";
    }
    /**
     * 重置参数，接收每帧lidar数据都要重置这些参数
    */
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}
        /**
     * 订阅原始imu数据
     * 1、imu原始测量数据转换到lidar系，加速度、角速度、RPY
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {   
        // imu原始测量数据转换到lidar系，加速度、角速度、RPY
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
         // 上锁，添加数据的时候队列不可用
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }
    /**
     * 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿？　这里没有预积分呀
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }
    /**
     * 订阅原始lidar数据
     * 1、添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
     * 2、当前帧起止时刻对应的imu数据、imu里程计数据处理
     *   imu数据：
     *   1) 遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     *   2) 用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
     *   imu里程计数据：
     *   1) 遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
     *   2) 用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
     * 3、当前帧激光点云运动畸变校正
     *   1) 检查激光点距离、扫描线是否合规
     *   2) 激光运动畸变校正，保存激光点
     * 4、提取有效激光点，存extractedCloud
     * 5、发布当前帧校正后点云，有效点
     * 6、重置参数，接收每帧lidar数据都要重置这些参数
    */
    void scanHandler(const sensor_msgs::LaserScanConstPtr& laserCloudMsg)//请收起你的多线程思维，这个函数里面全部是按顺序执行的
    {// 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
    //这里缓存了点云，同时这里也实现了ros->pcl的转换
        //将laserCloudMsg存起来
        
        if(!cacheScan(laserCloudMsg)){
            return ;
        }
        int seq=laserCloudMsg->header.seq;
        //logname.open("/home/shen/USTC/all_SLAM/slio_sam_ws/log/LM_log"+to_string(seq)+".txt");
        timeLaserInfoStamp = laserCloudMsg->header.stamp;
#if XMLLOG
        frame = new TiXmlElement("frame");
        imuOdomXml = new TiXmlElement("imuOdom");
        doc = new TiXmlDocument();
        doc->LinkEndChild(frame);
        frame ->LinkEndChild(imuOdomXml);
        frame->SetAttribute("seq",to_string(seq));
        stamp = new TiXmlElement("stamp");
        frame->LinkEndChild(stamp);
        stamp->SetAttribute("sec",to_string(timeLaserInfoStamp.sec));
        stamp->SetAttribute("nsec",to_string(timeLaserInfoStamp.nsec));
        TiXmlElement *o_scan = new TiXmlElement("orignal_scan");
        frame->LinkEndChild(o_scan);
        int indexx=0;
        for(auto opoint:*laserCloudIn){
            TiXmlElement *point = new TiXmlElement("point");
            o_scan->LinkEndChild(point);
            point->SetAttribute("index",indexx++);
            point->SetAttribute("x", convertToString(opoint.x));
            point->SetAttribute("y",convertToString(opoint.y));
            point->SetAttribute("z",convertToString(opoint.z));
            point->SetAttribute("intensity",convertToString(opoint.intensity));
        }
#endif
        // if (!cachePointCloud(tempPtr))
        //     return;
        // 当前帧起止时刻对应的imu数据、imu里程计数据处理
        if (!deskewInfo())
            return;
         // 当前帧激光点云运动畸变校正
        // 1、检查激光点距离、扫描线是否合规
        // 2、激光运动畸变校正，保存激光点
        projectPointCloud();
        //fullCloud
#if XMLLOG
        TiXmlElement *d_scan = new TiXmlElement("deskew_scan");
        frame->LinkEndChild(d_scan);
        indexx=0;
        for(auto dpoint:*fullCloud){
            TiXmlElement *point = new TiXmlElement("point");
            d_scan->LinkEndChild(point);
            point->SetAttribute("index",indexx++);
            point->SetAttribute("x", convertToString(dpoint.x));
            point->SetAttribute("y",convertToString(dpoint.y));
            point->SetAttribute("z",convertToString(dpoint.z));
            point->SetAttribute("intensity",convertToString(dpoint.intensity));
        }

        cout<<"保存xml与否,laserProjection:"<<doc->SaveFile("/home/shen/USTC/all_SLAM/slio_sam_ws/log/deskew_log"+to_string(seq)+".xml")<<endl;
        doc->Clear();
#endif
         // 提取有效激光点，存extractedCloud
        cloudExtraction();
         // 发布当前帧校正后点云，有效点
         //将extractedCloud存起来
        publishClouds();
        // 重置参数，接收每帧lidar数据都要重置这些参数
        resetParameters();
    }
    bool cacheScan(const sensor_msgs::LaserScanConstPtr& laserScanMsg){
#if PROCESS 
        ROS_INFO("--------点云预处理-start--------");
#endif
        scanQueue.push_back(*laserScanMsg);
        if(scanQueue.size()<2) return false;
        currentScanMsg = std::move(scanQueue.front()); 
        scanQueue.pop_front();
        PointXYZIT newPoint;//这里如实记录数据就好，点云去畸变同时也可以处理这种scan不水平的问题
        newPoint.z=0;
        double newPointAngle;
        int beamNum = currentScanMsg.ranges.size();
        //ROS_INFO("原始扫描点数量:%d",currentScanMsg.ranges.size());
        laserCloudIn->clear();
        for(int i=0;i<beamNum;++i){
            if(!std::isfinite(currentScanMsg.ranges[i])) //如果是非法值就不管它
                continue;
            newPointAngle   =  currentScanMsg.angle_min+currentScanMsg.angle_increment*i;
            newPoint.x  =    currentScanMsg.ranges[i]*cos(newPointAngle);
            newPoint.y  =    currentScanMsg.ranges[i]*sin(newPointAngle);
            newPoint.z = 0;
            newPoint.intensity  =   currentScanMsg.intensities[i];
            newPoint.time = currentScanMsg.time_increment*i;
            laserCloudIn->push_back(newPoint);
            //std::cout<<currentScanMsg.ranges[i]<<" "<<endl;
            //std::cout<<"x:"<<newPoint.x<<"  y:"<<newPoint.y<<"      ";
        }
        
        scanHeader = currentScanMsg.header;
        timeScanCur = scanHeader.stamp.toSec();
        timeScanEnd = timeScanCur+laserCloudIn->points.back().time;//不用担心，laserCloudln 每帧结束后都会重新清零的。
        deskewFlag = 1;//是否有时间通道
        //ROS_INFO("转化成pcl点云后:%d, liderFrame:%s",laserCloudIn->size(),lidarFrame.c_str());
        
        //publishCloud(pubLaserExtract, laserCloudIn, scanHeader.stamp, lidarFrame);
        return true;
    }
    /**
     * 当前帧起止时刻对应的imu数据、imu里程计数据处理
    */
    bool deskewInfo()
    {   
#if PROCESS 
        ROS_INFO("deskewInfo");
#endif 
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

       // 要求imu数据包含激光数据，否则不往下处理了
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }
        // 当前帧对应imu数据处理
        // 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
        // 2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
        // 注：imu数据都已经转换到lidar系下了
        
        imuDeskewInfo();
         // 当前帧对应imu里程计处理
        // 1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
        // 2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
        // 注：imu数据都已经转换到lidar系下了
        odomDeskewInfo();
#if PROCESS_IMU 
        cout<<"cloudInfo.imuAvailable:"<<cloudInfo.imuAvailable<<"  cloudInfo.odomAvailable:"<<cloudInfo.odomAvailable<<endl; 
#endif
        return true;
    }
    /**
     * 当前帧对应imu数据处理
     * 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     * 2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
     * 注：imu数据都已经转换到lidar系下了
    */
    void imuDeskewInfo()
    {
#if PROCESS 
        ROS_INFO("imuDeskewInfo");
#endif
        cloudInfo.imuAvailable = false;
           // 从imu队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;
        // 遍历当前激光帧起止时刻（前后扩展0.01s）之间的imu数据
        for (int i = 0; i < (int)imuQueue.size(); ++i)//注意这里是不会弹出的，所以放心遍历
        {   
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

          // 从imuMsg提取imu姿态角RPY给cloudInfo，作为当前lidar帧初始姿态角,我们这个imu没有提供rpy
          
            if (currentImuTime <= timeScanCur){
                
                //ROS_INFO("imuRPY2rosRPY前: R=%f P=%f Y=%f",cloudInfo.imuRollInit,cloudInfo.imuPitchInit,cloudInfo.imuYawInit);
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
                //ROS_INFO("imuRPY2rosRPY后: R=%f P=%f Y=%f",cloudInfo.imuRollInit,cloudInfo.imuPitchInit,cloudInfo.imuYawInit);
            }//这里应该为空
                
            //这个函数假设了imu里面有姿态信息，但是我们的那个玩意应该是没有的，这里得换个写法，
            if (currentImuTime > timeScanEnd + 0.01)
                break;
             //上面这一段的意思是取scantime-0.01 到　scantime的时间的imu信息求旋转作为当前scan的初始姿态   ，
             //也就是用当前帧附近(但小于当前帧)的imu信息得到当前帧的初始角度
            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }//这个结束后imuPointerCur==1

            // 提取imu角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // imu帧间时差
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];//第一帧imu，timeDiff等于零
             // 当前时刻旋转角 = 前一时刻旋转角 + 角速度 * 时差
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;//第一轮结束imuPointerCur==2,imuRotX[0] inuRotX[1] 都是０，后面开始就不一样了
            //imuTime里面存放的旋转是从0开始的
        }

        --imuPointerCur;//一开始是为了实现前后之差的，所以imuPointerCur作为下标比实际值大了一，所以要减回来
        // 没有合规的imu数据
        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }
/**
 * 当前帧对应imu里程计处理
 * 1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
 * 2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
 * 注：imu数据都已经转换到lidar系下了
*/
    void odomDeskewInfo()
    {
#if PROCESS 
        ROS_INFO("odomDeskewInfo");
#endif
        cloudInfo.odomAvailable = false;
        // 从imu里程计队列中删除当前激光帧0.01s前面时刻的imu数据
        
        while (!odomQueue.empty())
        {
            nav_msgs::Odometry thisOdom =  odomQueue.front();
            double prestamp=thisOdom.header.stamp.toSec();
            
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.020){
#if XMLLOG
                TiXmlElement *imuPose = new TiXmlElement("imuPose");
                imuOdomXml->LinkEndChild(imuPose);
                imuPose->SetAttribute("imuStamp",convertToString(prestamp));
                imuPose->SetAttribute("isScan",0);
                imuPose->SetAttribute("qx",convertToString(thisOdom.pose.pose.orientation.x));
                imuPose->SetAttribute("qy",convertToString(thisOdom.pose.pose.orientation.y));
                imuPose->SetAttribute("qz",convertToString(thisOdom.pose.pose.orientation.z));
                imuPose->SetAttribute("qw",convertToString(thisOdom.pose.pose.orientation.w));
                imuPose->SetAttribute("x",convertToString(thisOdom.pose.pose.position.x));
                imuPose->SetAttribute("y",convertToString(thisOdom.pose.pose.position.y));
                imuPose->SetAttribute("z",convertToString(thisOdom.pose.pose.position.z));
#endif 
                odomQueue.pop_front();
#if XMLLOG
                imuPose->SetAttribute("dt",convertToString(prestamp-odomQueue.front().header.stamp.toSec()));
#endif 
            }    
            else
                break;
            
            //cout<<"deltaT为:"<<prestamp-odomQueue.front().header.stamp.toSec()<<endl;
        }

        if (odomQueue.empty())
        {
            //cout<<"这里挂掉的1"<<endl;
             return;
        }
        
         // 要求必须有当前激光帧时刻之前的imu里程计数据
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
        {
            //cout<<"这里挂掉的2"<<endl;
             return;
        }

        // 提取当前激光帧起始时刻的imu里程计
        nav_msgs::Odometry startOdomMsg;//这个频率和imu一样

        for (int i = 0; i < (int)odomQueue.size(); ++i)//寻找启始里程记
        {
            startOdomMsg = odomQueue[i];
#if XMLLOG
            double prestamp=startOdomMsg.header.stamp.toSec();
            TiXmlElement *imuPose = new TiXmlElement("imuPose");
            imuOdomXml->LinkEndChild(imuPose);
            imuPose->SetAttribute("imuStamp",convertToString(prestamp));
            if(ROS_TIME(&startOdomMsg) < timeScanCur)    imuPose->SetAttribute("isScan",0);
            else imuPose->SetAttribute("isScan",1);
            imuPose->SetAttribute("qx",convertToString(startOdomMsg.pose.pose.orientation.x));
            imuPose->SetAttribute("qy",convertToString(startOdomMsg.pose.pose.orientation.y));
            imuPose->SetAttribute("qz",convertToString(startOdomMsg.pose.pose.orientation.z));
            imuPose->SetAttribute("qw",convertToString(startOdomMsg.pose.pose.orientation.w));
            imuPose->SetAttribute("x",convertToString(startOdomMsg.pose.pose.position.x));
            imuPose->SetAttribute("y",convertToString(startOdomMsg.pose.pose.position.y));
            imuPose->SetAttribute("z",convertToString(startOdomMsg.pose.pose.position.z));
             if(i+1<(int)odomQueue.size())   imuPose->SetAttribute("dt",convertToString(prestamp-odomQueue[i+1].header.stamp.toSec()));
#endif
            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }
#if XMLLOG
        TiXmlElement *PoseXml = new TiXmlElement("Pose");
        frame->LinkEndChild(PoseXml);
        PoseXml->SetAttribute("qx",convertToString(startOdomMsg.pose.pose.orientation.x));
        PoseXml->SetAttribute("qy",convertToString(startOdomMsg.pose.pose.orientation.y));
        PoseXml->SetAttribute("qz",convertToString(startOdomMsg.pose.pose.orientation.z));
        PoseXml->SetAttribute("qw",convertToString(startOdomMsg.pose.pose.orientation.w));
        PoseXml->SetAttribute("x",convertToString(startOdomMsg.pose.pose.position.x));
        PoseXml->SetAttribute("y",convertToString(startOdomMsg.pose.pose.position.y));
        PoseXml->SetAttribute("z",convertToString(startOdomMsg.pose.pose.position.z));
#endif
        // 提取imu里程计姿态角
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        //注意这里的imu里程记也是在激光帧的基础上的，所以可以不用担心它误差累计的问题，影响比较小
       // 用当前激光帧起始时刻的imu里程计，初始化lidar位姿，后面用于mapOptmization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll    = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;
        cloudInfo.odomAvailable = true;
        // get end odometry at the end of the scan
        odomDeskewFlag = false;
        // 如果当前激光帧结束时刻之后没有imu里程计数据，返回
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)//timeScanEnd是当前激光帧结束的时刻，这个我们可能没有，需要人为添加
        {
            //cout<<"这里挂掉的3"<<endl;
             return;
        }
        // 提取当前激光帧结束时刻的imu里程计
        nav_msgs::Odometry endOdomMsg;
        //还是用遍历的方法，真的是
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];
#if XMLLOG
            double prestamp=endOdomMsg.header.stamp.toSec();
            TiXmlElement *imuPose = new TiXmlElement("imuPose");
            imuOdomXml->LinkEndChild(imuPose);
            imuPose->SetAttribute("imuStamp",convertToString(prestamp));
            if(ROS_TIME(&endOdomMsg) < timeScanEnd)    imuPose->SetAttribute("isScan",2);
            else imuPose->SetAttribute("isScan",3);
            imuPose->SetAttribute("qx",convertToString(startOdomMsg.pose.pose.orientation.x));
            imuPose->SetAttribute("qy",convertToString(startOdomMsg.pose.pose.orientation.y));
            imuPose->SetAttribute("qz",convertToString(startOdomMsg.pose.pose.orientation.z));
            imuPose->SetAttribute("qw",convertToString(startOdomMsg.pose.pose.orientation.w));
            imuPose->SetAttribute("x",convertToString(startOdomMsg.pose.pose.position.x));
            imuPose->SetAttribute("y",convertToString(startOdomMsg.pose.pose.position.y));
            imuPose->SetAttribute("z",convertToString(startOdomMsg.pose.pose.position.z));
            if(i+1<(int)odomQueue.size())   imuPose->SetAttribute("dt",convertToString(prestamp-odomQueue[i+1].header.stamp.toSec()));

#endif
            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        // 如果起止时刻对应imu里程计的方差不等，返回，这是什么意思呀?这个可能的意思是，随着时间的推移，imu逐渐不可信了，方差一般来说都是很小的，而int是取整所以这里的不等其实已经差了很多
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
        {
            //cout<<"这里挂掉的4"<<endl;
             return;
        }
        //这里return了怎么办，然道就不用它了吗?
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 起止时刻imu里程计的相对变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        // 相对变换，提取增量平移、旋转（欧拉角）
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);
        //去畸变的核心是得到odomIncreX/Y  /Z
        odomDeskewFlag = true;
    }
/**
 * 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
*/
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        //imuPointerCur,代表当前帧激光点云数量，这个在deskewInfo()这个函数里被赋值的。
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)//寻找恰好大于pointTime的imuTime对应的下标imuPointerFront
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }//意外情况就是imuPointFront==imuPointerCur 也就是没有找到
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }
/**
 * 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量）
*/
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {// 如果传感器移动速度较慢，例如人行走的速度，那么可以认为激光在一帧时间范围内，平移量小到可以忽略不计
        *posXCur = 0; *posYCur = 0; *posZCur = 0;
        if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
             return;
        float ratio = relTime / (timeScanEnd - timeScanCur);//这里假设运动是匀速的
         *posXCur = ratio * odomIncreX;
         *posYCur = ratio * odomIncreY;
         *posZCur = ratio * odomIncreZ;
    }
/**
 * 激光运动畸变校正
 * 利用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量，进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
*/
    PointType deskewPoint(PointType *point, double relTime)
    {
        //return *point;//不进行去畸变 看看会不会有影响
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point; //如果没有imu信息就直接返回原始数据，不进行去畸变
        // relTime是当前激光点相对于激光帧起始时刻的时间，pointTime则是当前激光点的时间戳
        double pointTime = timeScanCur + relTime;
         // 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        // 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量）
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        // 第一个点的位姿增量（0），求逆
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

         // 当前时刻激光点与第一个激光点的位姿变换
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;
         // 当前激光点在第一个激光点坐标系下的坐标
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }
    /**
     * 当前帧激光点云运动畸变校正
     * 1、检查激光点距离、扫描线是否合规
     * 2、激光运动畸变校正，保存激光点
    */
    void projectPointCloud()//出了问题要定位这里
    {
        //ROS_INFO("projectPointCloud,原始雷达点云数量:%d",laserCloudIn->points.size());
        int cloudSize = laserCloudIn->points.size();//这里可以怎么做的假设就是laserCloudln 刚好对应一个scan

        // 遍历当前帧激光点云
        for (int i = 0; i < cloudSize; ++i)
        {   // pcl格式
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // 距离检查
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            //将thisPoint放入
            //关闭点云去畸变看看
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            
            // 矩阵存激光点的距离
            rangeMat.at<float>(0, i) = range;
            //cout<<"rangeMat"<<i<<":"<<rangeMat.at<float>(0, i) <<" ";
            //thisPoint.z=0;//这里为了简化后面的优化，假定机器人在平面上移动
            fullCloud->points[i] = thisPoint;//去畸变后点云，不过这里还没有去除不规范点
        }
    }
    /**
     * 提取有效激光点，存extractedCloud
    */
    void cloudExtraction()
    {
#if PROCESS 
        ROS_INFO("cloudExtraction()");
#endif
        // 有效激光点数量
        int count = 0;
        // 记录每根扫描线起始第5个激光点在一维数组中的索引
        cloudInfo.startRingIndex[0] = count - 1 + 5;
        for (int j = 0; j < Horizon_SCAN; ++j)
        {   // 有效激光点
            if (rangeMat.at<float>(0,j) != FLT_MAX)
            {
                    // 记录激光点对应的Horizon_SCAN方向上的索引
                cloudInfo.pointColInd[count] = j;
                // 激光点距离
                cloudInfo.pointRange[count] = rangeMat.at<float>(0,j);
              //  cout<<"pointRange"<<j<<":"<<rangeMat.at<float>(0,j) <<endl;
                    // 加入有效激光点
                extractedCloud->push_back(fullCloud->points[j + 0*Horizon_SCAN]);
                // 有效激光点数量
                ++count;
            }
        }
        // 记录每根扫描线倒数第5个激光点在一维数组中的索引
        cloudInfo.endRingIndex[0] = count -1 - 5;
    }
    
/**
 * 发布当前帧校正后点云，有效点
*/
    void publishClouds()
    {
#if PROCESS
    ROS_INFO("publishClouds()");
#endif
        //ROS_INFO("发布当前帧校正后点云，有效点数量:%d",extractedCloud->size());
        cloudInfo.header = scanHeader;
        //参数从左到右依次是函数指针，提取的点云，原始scan头信息，雷达frame_id
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, scanHeader.stamp, lidarFrame);



        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    setlocale(LC_CTYPE,"zh_CN.utf8");
    ros::init(argc, argv, "slio_sam");

    ImageProjection IP;
    ROS_INFO("\033[1;32m---->启动点云去畸变节点.\033[0m");

    ros::MultiThreadedSpinner spinner(3);//多线程，这里用了三个线程，可能是里面有三个subscript
    spinner.spin();
    
    return 0;
}
