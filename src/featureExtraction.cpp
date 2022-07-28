#include "../include/utility.h"
#include "slio_sam/cloud_info.h"
#define PROCESS 0
struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    slio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction()
    {
        //ROS_INFO("FeatureExtraction:");
        subLaserCloudInfo = nh.subscribe<slio_sam::cloud_info>("slio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<slio_sam::cloud_info> ("slio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("slio_sam/feature/cloud_surface", 1);
        //ROS_INFO("initializationValue");
        initializationValue();
    }

    void initializationValue()
    {
        //ROS_INFO("initializationValue_start");
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);//降采样大小

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
        //ROS_INFO("initializationValue_end");
    }

    void laserCloudInfoHandler(const slio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // 这里相当于单独把它点信息给拿出来了
        //ROS_INFO("featureExtraction: 去畸变后点数量：%d",extractedCloud->size());
        //计算曲率
        //ROS_INFO("计算曲率\n，去畸变后点云数量:%d,扫描点数量:%d",extractedCloud->size(),cloudInfo.pointRange.size());
        calculateSmoothness();
        //标记异常点
        //ROS_INFO("记录异常点");
        markOccludedPoints();
        //提取特征点并且对线特征降采样，这里线特征可以考虑换一个方式，不要直接进行降采样，
        //能不能换一个方式降采样，按线分配特征点，有的线长有的线短,让每个线都有均匀分布特征点。
        //ROS_INFO("提取特征点");
        extractFeatures();
        //ROS_INFO("发布特征点");
        publishFeatureCloud();
        //ROS_INFO("--------点云预处理-end-------");
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();//这个是提取后的，其位置和原始的点云位置不一样了,不过couldinfo.pointColInd存储了下标
        for (int i = 5; i < cloudSize - 5; i++)
        {   //cout<<"cInfo.Range"<<i<<":"<<cloudInfo.pointRange[i]<<endl;
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            // cout<<"cloudCurvature"<<i<<":"<<cloudCurvature[i]<<" ";
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            //获取有效点之间的下标间隔，理论上应该是１，但是因为一些无效点的存在结果会大于１
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));
              //如果中间间隔不是很大，这里主要是预防两个有效点之间无效点过多。
            if (columnDiff < 10){
                //相邻的两帧间隔深度差大于0.3说明遇到遮挡点了，标记为已经选择过，下次就不会再对这些点进行特征点提取了。
                if (depth1 - depth2 > 0.05){//０.3 这些参数都是经过计算得到的，自己也要好好算算，这个是不是有点大了应该改成0.1
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.05){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // 获取当前点相邻之间的深度差
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));
            //如果相邻深度都较大，则认为当前点为平行光束，并标记为已经选择过
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    // edgeThreshold为0.1，正圆的曲率为0
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20){
                            cloudLabel[ind] = 1;//标记点的类型，如果判断是角点，就会被标记为１
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }
                        // 防止特征点聚集，将ind及其前后各5个点标记，不做特征点提取
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= mask; l++)
                        {   // 每个点index之间的差值。附近点都是有效点的情况下，相邻点间的索引只差１
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            // 附近有无效点，或者是每条线的起点和终点的部分
                            if (columnDiff > 10)//这个是用来
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -mask; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    //surfThreshold=0.1
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0)//edgeThreshold 修改为0
                    {
                        //特征点点标记，如果是面点，就会给标记为-1,标记完成后，表示特征点不在提取
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;
                        //同样的为防止特征点聚集，将ind及其前后各5个点标记，不做特征点提取
                        for (int l = 1; l <= mask; l++) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -mask; l--) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                //根据标记获取平面点
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }
           
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);//20cm,面特征降采样
            downSizeFilter.filter(*surfaceCloudScanDS);
            *surfaceCloud += *surfaceCloudScanDS;
        }
        //ROS_INFO("featureExtration 角特征点数量%d,平面特征点数量%d",cornerCloud->size(),surfaceCloud->size());
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        //注对于单线激光雷达来说，线特征就是点特征，面特征就是线特征。
        //ROS_INFO("publishFeatureCloud 角特征点数量:%d,面特征点数量:%d",cornerCloud->size(),surfaceCloud->size());
        cloudInfo.cloud_corner  = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    setlocale(LC_CTYPE,"zh_CN.utf8");
    ros::init(argc, argv, "slio_sam");

    FeatureExtraction FE;

    //ROS_INFO("\033[1;32m----> Feature Extraction Started------------.\033[0m");
    ROS_INFO("\033[1;32m---->启动feature提取.\033[0m");
    ros::spin();

    return 0;
}