#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <ORB_SLAM2/include/System.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <pcl/correspondence.h>
#include <pcl_ros/point_cloud.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

using namespace std;
class FramePoint
{
  public:
  Eigen::Matrix4f transMatrix = Eigen::Matrix4f::Identity(4,4);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZINormal>};
};
vector<FramePoint> framepoint;
void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orbslam2_stereo_bridge_node");

    // Get rosparams
    ros::NodeHandle pnh("~");
    bool is_image_compressed;
    bool is_image_color = true;
    bool online_rectification;
    int recollection = 30;
    string vocab_path, vslam_config_path, image_topic_name0, image_topic_name1;
    pnh.getParam("vocab_path", vocab_path);
    pnh.getParam("vslam_config_path", vslam_config_path);
    pnh.getParam("image_topic_name0", image_topic_name0);
    pnh.getParam("image_topic_name1", image_topic_name1);
    pnh.getParam("is_image_compressed", is_image_compressed);
    pnh.getParam("is_image_color", is_image_color);
    pnh.getParam("keyframe_recollection", recollection);
    pnh.getParam("online_rectification", online_rectification);
    ROS_INFO("vocab_path: %s, vslam_config_path: %s, image_topic_name: %s, is_image_compressed: %d",
        vocab_path.c_str(), vslam_config_path.c_str(), image_topic_name0.c_str(), is_image_compressed);

    // Setup subscriber
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    
    // Setup image subscriber
    message_filters::Subscriber<sensor_msgs::CompressedImage> infra1_image_subscriber(nh, image_topic_name0, 1);
    message_filters::Subscriber<sensor_msgs::CompressedImage> infra2_image_subscriber(nh, image_topic_name1, 1);
    message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> syncronizer(infra1_image_subscriber, infra2_image_subscriber, 10);

    ros::Time subscribed_stamp;
    cv::Mat subscribed_image0, subscribed_image1;

    if (is_image_compressed) {
        auto image_callback = [is_image_color, &subscribed_image0, &subscribed_image1, &subscribed_stamp](const sensor_msgs::CompressedImageConstPtr& image0, const sensor_msgs::CompressedImageConstPtr& image1) -> void {
        subscribed_image0 = cv::imdecode(cv::Mat(image0->data), is_image_color ? 1 : 0 /* '1': bgr, '0': gray*/);
        subscribed_image1 = cv::imdecode(cv::Mat(image1->data), is_image_color ? 1 : 0 /* '1': bgr, '0': gray*/);
        subscribed_stamp = image0->header.stamp;
        };
        syncronizer.registerCallback(boost::bind<void>(image_callback, _1, _2));
    } else {
        std::cerr << "Error: Only compressed image is acceptable" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Setup publisher
    ros::Publisher vslam_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("iris/vslam_data", 1);
    image_transport::Publisher image_publisher = it.advertise("iris/processed_image", 5);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(vocab_path, vslam_config_path, ORB_SLAM2::System::STEREO, false);

    cv::Mat M1l,M2l,M1r,M2r;
    if(online_rectification) {
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(vslam_config_path, cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }
        
        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
    }

    std::chrono::system_clock::time_point m_start;
    ros::Rate loop_rate(30);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vslam_data(new pcl::PointCloud<pcl::PointXYZINormal>);
    
    // Start main loop
    ROS_INFO("start main loop.");
    while (ros::ok()) {
        if (!subscribed_image0.empty() && !subscribed_image1.empty()) {
            m_start = std::chrono::system_clock::now();  // start timer
            ros::Time process_stamp = subscribed_stamp;
            cv::Mat cur_pose;
            if(online_rectification) {
                cv::Mat imLeft, imRight;
                cv::remap(subscribed_image0,imLeft,M1l,M2l,cv::INTER_LINEAR);
                cv::remap(subscribed_image1,imRight,M1r,M2r,cv::INTER_LINEAR);
                cur_pose = SLAM.TrackStereo(imLeft, imRight, subscribed_stamp.toSec());
            }
            else {
                cur_pose = SLAM.TrackStereo(subscribed_image0, subscribed_image1, subscribed_stamp.toSec());
            }
            
            
            // process ORB_SLAM2
            Eigen::Matrix4f eigen_mat;
            if(!cur_pose.empty()) {
                cv::cv2eigen(cur_pose, eigen_mat);
                publishPose(eigen_mat.inverse(), "iris/vslam_pose");
            }
            if(SLAM.GetTrackingState() == ORB_SLAM2::Tracking::OK) {
                {
                    cv::Mat track_img;

                    drawKeypoints(subscribed_image0, SLAM.GetTrackedKeyPointsUn(), track_img, cv::Scalar(0,255,0));
                    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_img).toImageMsg();
                    image_publisher.publish(msg);
                }
                vector<ORB_SLAM2::MapPoint*> trackedPoint = SLAM.GetTrackedMapPoints();
                cout << trackedPoint.size() << endl;
                if (trackedPoint.empty()) {
                    ROS_INFO("No Tracked Point");
                }
                else {
                    vslam_data->clear();
                    const size_t N = trackedPoint.size();
                    for(size_t i=0; i<N; i++)
                    {
                        if(trackedPoint[i]) {
                            cv::Mat point = trackedPoint[i]->GetWorldPos();
                            Eigen::Vector3f eigen_point;
                            cv::cv2eigen(point, eigen_point);
                            float dist = fabs((eigen_point - eigen_mat.inverse().block<3,1>(0,3)).norm());
                            if (dist < 30 && dist > 2) {
                                pcl::PointXYZINormal p;
                                p.x = static_cast<float>(point.at<float>(0));
                                p.y = static_cast<float>(point.at<float>(1));
                                p.z = static_cast<float>(point.at<float>(2));
                                float weight = static_cast<float>(1.0 - dist * 0.2);
                                p.intensity = weight;
                                vslam_data->push_back(p);
                            }
                        }
                    }
                    if (framepoint.size() == 0) {
                        FramePoint currentpoint;
                        currentpoint.cloud->points.clear();
                        currentpoint.cloud->points.assign(vslam_data->points.begin(),vslam_data->points.end()); 
                        currentpoint.transMatrix = eigen_mat.inverse();
                        framepoint.push_back(currentpoint);
                    }
                    else if (fabs((eigen_mat.inverse().block<3,1>(0,3) - framepoint.back().transMatrix.block<3,1>(0,3)).norm()) > 5) {
                        FramePoint currentpoint;
                        currentpoint.cloud->points.clear();
                        currentpoint.cloud->points.assign(vslam_data->points.begin(),vslam_data->points.end()); 
                        currentpoint.transMatrix = eigen_mat.inverse();
                        framepoint.push_back(currentpoint);
                        if(framepoint.size() > 4) 
                            framepoint.erase(framepoint.begin());
                    }
                    for (int i=0; i<framepoint.size(); i++) {
                        *vslam_data += *framepoint[i].cloud;
                    }

                    pcl_conversions::toPCL(process_stamp, vslam_data->header.stamp);
                    vslam_data->header.frame_id = "world";
                    vslam_publisher.publish(vslam_data);
                }
            }

            // Inform processing time
            std::stringstream ss;
            long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count();
            ss << "orb processing time= \033[35m"
                << time_ms
                << "\033[m ms";
            ROS_INFO("%s", ss.str().c_str());
        }
        // Reset input
        subscribed_image0 = cv::Mat();
        subscribed_image1 = cv::Mat();
        // Spin and wait
        loop_rate.sleep();
        ros::spinOnce();
    }
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("Trajectory.txt");

    ROS_INFO("Finalize orbslam2_bridge::bridge_node");

    return 0;
}

void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id)
{
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setFromOpenGLMatrix(T.cast<double>().eval().data());
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", child_frame_id));
}