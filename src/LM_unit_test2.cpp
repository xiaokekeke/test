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
#define LM2 1
#define PICK 0
#define GIVETMS 0
using namespace std;
pcl::PointCloud<PointType> pclmap ;
pcl::PointCloud<PointType> pclscan ;
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>);
pcl::KdTreeFLANN<PointType>::Ptr kdtreeFeatureFromScan(new pcl::KdTreeFLANN<PointType>);
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType> savelaserCloudOri;
pcl::PointCloud<PointType> savecoeffSel;
Eigen::Affine3f transPointAssociateToMap;
pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS(new pcl::PointCloud<PointType>); 
std::vector<PointType> laserCloudOriCornerVec(1445);
std::vector<bool> laserCloudOriCornerFlag(1445);
std::vector<PointType> coeffSelCornerVec(1445);
visualization_msgs::Marker line_list;//这个是自己加的主要是为了显示提取的直线正确与否
visualization_msgs::Marker coeff_list;
cv::Mat matP;
double theta;
float transformTobeMapped[6];
int laserCloudSurfLastDSNum = 0;
bool isDegenerate = false;
double Tms0[6];//优化前还未与imu数据融合的初始值
double Tms_iter[100][6];
double chis[30];
double deltachis[30];
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
       
        if(strcmp(elem_name.c_str(),"Tms0")==0){
             cout<<elem_name<<endl;
#if GIVETMS
            Tms0[0]=-0.00117779;
            Tms0[1]=-0.00521441;
            Tms0[2]= 3.06854;
            Tms0[3]= 10.9365;
            Tms0[4]= -0.419833;
            Tms0[5]= -0.03748;
#else
            Tms0[0]=atof(elem->Attribute("rx"));
            Tms0[1]=atof(elem->Attribute("ry"));
            Tms0[2]=atof(elem->Attribute("rz"));
            Tms0[3]=atof(elem->Attribute("x"));
            Tms0[4]=atof(elem->Attribute("y"));
            Tms0[5]=atof(elem->Attribute("z"));
#endif
            //LM_可以用，但是得好好写，还有要高清楚预防退化的原理
            for(int i=0;i<6;++i){
                transformTobeMapped[i] = Tms0[i];
            }
            
        }
        //pcl::pointCloud<pointType> 待会儿用publishCloud函数发送
        if(strcmp(elem_name.c_str(),"Map")==0){
             cout<<elem_name<<endl;
             for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
                 PointType mappoint;
                 mappoint.x = atof(elem2->Attribute("x"));
                 mappoint.y =atof(elem2->Attribute("y"));
                 mappoint.z =atof(elem2->Attribute("z"));
                 mappoint.intensity =atof(elem2->Attribute("intensity"));
                 pclmap.push_back(mappoint);
                 laserCloudSurfFromMapDS->push_back(mappoint);
             }
        }
        
        if(strcmp(elem_name.c_str(),"Scan")==0){
             cout<<elem_name<<endl;
             for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
                 PointType scanpoint;
                 scanpoint.x = atof(elem2->Attribute("x"));
                 scanpoint.y =atof(elem2->Attribute("y"));
                 scanpoint.z =atof(elem2->Attribute("z"));
                 scanpoint.intensity =atof(elem2->Attribute("intensity"));
                 pclscan.push_back(scanpoint);
                 laserCloudSurfLastDS->push_back(scanpoint);
             }
             laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();
        }
    }
    doc.Clear();
}
Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
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
void pointAssociateToMap(PointType const * const pi, PointType * const po)
{  
    po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
    po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
    po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
    po->intensity = pi->intensity;
}
void updatePointAssociateToMap()
{
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}
void cornerOptimization()
    {// 更新当前位姿与地图间位姿变换,这个是什么意思?
        //ROS_INFO("遍历点云, 构建点到直线的约束");
        updatePointAssociateToMap();
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
             //cout<<"pointOri:"<<i<<" "<<pointOri.x<<" "<<pointOri.y<<" "<<pointOri.z<<" "<<pointOri.intensity<<endl;
             //cout<<"pointSel:"<<i<<" "<<pointSel.x<<" "<<pointSel.y<<" "<<pointSel.z<<" "<<pointSel.intensity<<endl;
            
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
                    //cout<<"s:"<<s<<endl;
                    if (s > 0.1) {//角点以及角点到直线法线的向量
                        // 当前激光帧角点，加入匹配集合中
                        //cout<<"x"<<endl;
                        laserCloudOriCornerVec[i] = pointOri;//因为我们待会还是要采用线特征点的方法去计算，所以这里还是放在角点这里
                        //cout<<"y"<<endl;
                        // 角点的参数，
                        coeffSelCornerVec[i] = coeff;
                        //cout<<"z"<<endl;
                        laserCloudOriCornerFlag[i] = true;

                    }
                }
            }
            //cout<<"6666"<<endl;
        }

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
}
void recoverStatus(){
    *laserCloudOri = savelaserCloudOri;
    *coeffSel = savecoeffSel;
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
    cornerOptimization();
    combineOptimizationCoeffs();
    double current_chi = getChi();
    recoverStatus();
    return last_chi - current_chi;
}
int LMOptimization(int iterCount){//这个是一点点改的
    cout<<"-------iter:"<<iterCount<<"----------------------"<<endl;
    float sx = sin(transformTobeMapped[0]);
    float cx = cos(transformTobeMapped[0]);
    float sy = sin(transformTobeMapped[1]);
    float cy = cos(transformTobeMapped[1]);
    float sz = sin(transformTobeMapped[2]);
    float cz = cos(transformTobeMapped[2]);
    int laserCloudSelNum = laserCloudOri->size();
    cout<<"laserCloudSelNum:"<<laserCloudSelNum<<endl;
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
        }
        int msize=0;
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
        cout<<"matAtA:"<<endl;
        for(int i=0;i<6;++i){
            for(int j=0;j<6;++j){
                cout<<matAtA.at<float>(i,j)<<" ";
            }
            cout<<endl;
        }
        cout<<"matAtB:"<<endl;
        for(int i=0;i<6;++i){
            cout<<matAtB.at<float>(i,0);
            cout<<endl;
        }

        cv::Mat matE(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
        cv::eigen(matAtA, matE, matV);//matE为特征值 matV 为特征向量
        cout<<"matE;"<<endl;
        for(int i=0;i<6;++i){
            cout<<matE.at<float>(i,0);
            cout<<endl;
        }
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
            cout<<"matP:"<<endl;
            for(int i=0;i<6;++i){
                for(int j=0;j<6;++j){
                    cout<<matP.at<float>(i,j)<<" ";
                }
                cout<<endl;
            }
        }

        if (isDegenerate)//退化
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        if(iterCount ==0 ) theta = 1;
        //matX=matAtB*0.001;
        matX*=theta;//直接用梯度下降法，看看会有什么结果
        cout<<"matX;"<<endl;
        for(int i=0;i<6;++i){
            cout<<matX.at<float>(i,0);
            cout<<endl;
        }
        cout<<"tms_start:";
        for(int i=0;i<6;++i) cout<<transformTobeMapped[i]<<" ";
        cout<<endl;
        for(int i = 0;i<6;++i)
            transformTobeMapped[i] += matX.at<float>(i, 0);
        cout<<"tms_end:";
        for(int i=0;i<6;++i) cout<<transformTobeMapped[i]<<" ";
        cout<<endl;
        for(int i=0;i<6;++i)    Tms_iter[iterCount][i] = transformTobeMapped[i];
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        //logname<<iterCount<<"_下降量: R:"<<deltaR<<" T:"<<deltaT<<endl;
        //意思是如果修改的角度的平方小于1度^2，位移小于1cm^2
        cv::transpose(matB, matBt);
        matBtB=matBt*matB;

        double chi = 0.5*matBtB.at<float>(0,0); 
        chis[iterCount] = chi;
        double deltachi=testTms();//返回的结果三last_chi - current_chi
        deltachis[iterCount] =deltachi; 
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
            return 2;
        }
}
    void scan2MapOptimization()
    {// 构建点到平面、点到直线的残差, 用高斯牛顿法进行优化
        //ROS_INFO("---------------------scan2MapOptimization_start---------------");
        // 降采样后的角点与面点的数量要同时大于阈值才进行优化
        // surfFeatureMinValidNum 100
        cout<<"laserCloudSurfLastDSNum:"<<laserCloudSurfLastDSNum<<endl;
# if PICK
        bool peakflags[1500]={false};
        kdtreeFeatureFromScan->setInputCloud(laserCloudSurfLastDS);
         pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS_temp(new pcl::PointCloud<PointType>);        
         //cout<<"xxxxxxxxxxxx"<<endl;   
        for(int i=0;i<laserCloudSurfLastDSNum;++i){
            //cout<<"xxxxxxxxxxxx"<<endl;   
            PointType pointOri;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            pointOri = laserCloudSurfLastDS->points[i];
            kdtreeFeatureFromScan->nearestKSearch(pointOri, 5, pointSearchInd, pointSearchSqDis);
            if(pointSearchSqDis[1]<0.05){
                laserCloudSurfLastDS_temp->push_back(pointOri);
            }
        }
        *laserCloudSurfLastDS = *laserCloudSurfLastDS_temp;
        pclscan = *laserCloudSurfLastDS_temp;
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        cout<<"after_pick_laserCloudSurfLastDSNum:"<<laserCloudSurfLastDSNum<<endl;
# endif
        if (laserCloudSurfLastDSNum > 50)
        {//接下来我们要做的就是，将点到线距离改成点到点距离，点到面距离，改成点到线距离。
        //laserCloudSurfFromMap存放的是整个地图降采用的结果
            kdtreeCornerFromMap->setInputCloud(laserCloudSurfFromMapDS);
            int LMflag = 0;
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                cout<<"iterCount:"<<iterCount<<" "<<"LMflag:"<<LMflag<<endl;
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
                        
                        cornerOptimization();
                        combineOptimizationCoeffs();
                        LMflag=LMOptimization(iterCount);
                        
                    }else if(LMflag == 1){//如果迭代导致结果误差变大,回滚，且不重新提取特征
                        LMflag=LMOptimization(iterCount);
                    }
                    if (LMflag == 2)//如果迭代成功
                        break;            
                #endif  
                cout<<"tms_iter:";
                for(int i = 0;i<6;++i){
                    
                    cout<<transformTobeMapped[i]<<" ";
                    itertotal = iterCount+1;
                }
                cout<<endl;
            }
        } 
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
    ros::Publisher mapPublisher = nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/map",1);    
    ros::Publisher iterFramePublisher  =    nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/iterFrame",1);
    ros::Publisher originFramePublisher =  nh.advertise<sensor_msgs::PointCloud2>("/LM_unit_test/originFrame",1);
    ros::Publisher coeffPublisher = nh.advertise<visualization_msgs::Marker>("LM_unit_test/coeff",1);
    ros::Publisher linePublisher = nh.advertise<visualization_msgs::Marker>("LM_unit_test/line",1);
    ros::Publisher pointPublisher = nh.advertise<visualization_msgs::Marker>("LM_unit_test/point",1);
    ros::Publisher posePublisher = nh.advertise<geometry_msgs::PoseStamped>("/LM_unit_test/Pose",1);
    string filename;
    nh.param<std::string>("/slio_sam/unit_test_path", filename, "./log/LM_log90.xml");
    loadXML(filename);
    scan2MapOptimization();

    //先把map数据弄出来 ，接下来每次循环都发布
    publishCloud(mapPublisher,pclmap,"world");
    //再把原始frame弄出来，接下来巡回每次都发布
    //点云开始迭代的启始frame
    pcl::PointCloud<PointType> start0frame;
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id="world";
    Eigen::Quaterniond qms;
    Eigen::Vector3d tms;
    transPoints(pclscan,start0frame,Tms0);
    ros::Rate loop_rate(1);//1hz
    int k=0;
    while(ros::ok()){
        publishCloud(mapPublisher,pclmap,"world");
        publishCloud(originFramePublisher,pclscan,"world");
        pcl::PointCloud<PointType> iterframe;
        transPoints(pclscan,iterframe,Tms_iter[k%itertotal]);
        publishCloud(iterFramePublisher,iterframe,"world");
        se2SE(Tms_iter[k%itertotal],qms,tms);
        pose.pose.orientation.x=qms.x();
        pose.pose.orientation.y=qms.y();
        pose.pose.orientation.z=qms.z();
        pose.pose.orientation.w=qms.w();
        pose.pose.position.x=tms.x();
        pose.pose.position.y=tms.y();
        pose.pose.position.z=tms.z();
        posePublisher.publish(pose);
        ROS_INFO("iter_cout:%d  chi:%f  deltachi:%f",k%itertotal,chis[k%itertotal],deltachis[k%itertotal]);
        //每次提取line coeff point 可视化 
        //每次迭代后点云位置
        ++k;
        loop_rate.sleep();
    }
}