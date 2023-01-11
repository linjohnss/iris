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

using namespace std;

void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orbslam2_bridge_node");

    // Get rosparams
    ros::NodeHandle pnh("~");
    bool is_image_compressed;
    int recollection = 30;
    std::string vocab_path, vslam_config_path, image_topic_name;
    pnh.getParam("vocab_path", vocab_path);
    pnh.getParam("vslam_config_path", vslam_config_path);
    pnh.getParam("image_topic_name0", image_topic_name);
    pnh.getParam("is_image_compressed", is_image_compressed);
    pnh.getParam("keyframe_recollection", recollection);
    ROS_INFO("vocab_path: %s, vslam_config_path: %s, image_topic_name: %s, is_image_compressed: %d",
        vocab_path.c_str(), vslam_config_path.c_str(), image_topic_name.c_str(), is_image_compressed);

    // Setup subscriber
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ros::Time subscribed_stamp;
    cv::Mat subscribed_image;
    image_transport::TransportHints hints("raw");
    if (is_image_compressed) hints = image_transport::TransportHints("compressed");

    auto callback = [&subscribed_image, &subscribed_stamp](const sensor_msgs::ImageConstPtr& msg) -> void {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        subscribed_image = cv_ptr->image.clone();
        subscribed_stamp = msg->header.stamp;
    };

    image_transport::Subscriber image_subscriber = it.subscribe(image_topic_name, 5, callback, ros::VoidPtr(), hints);
    
    // Setup publisher
    ros::Publisher vslam_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("iris/vslam_data", 1);
    image_transport::Publisher image_publisher = it.advertise("iris/processed_image", 5);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(vocab_path, vslam_config_path, ORB_SLAM2::System::MONOCULAR, false);

    std::chrono::system_clock::time_point m_start;
    ros::Rate loop_rate(30);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vslam_data(new pcl::PointCloud<pcl::PointXYZINormal>);
    float accuracy = 0.5f;
    
    // Start main loop
    ROS_INFO("start main loop.");
    while (ros::ok()) {
        if (!subscribed_image.empty()) {
            m_start = std::chrono::system_clock::now();  // start timer
            ros::Time process_stamp = subscribed_stamp;

            // process ORB_SLAM2
            cv::Mat cur_pose = SLAM.TrackMonocular(subscribed_image,subscribed_stamp.toSec());
            if(!cur_pose.empty()) {
                Eigen::Matrix4f eigen_mat;
                cv::cv2eigen(cur_pose, eigen_mat);
                publishPose(eigen_mat.inverse(), "iris/vslam_pose");
            }

            if(SLAM.GetTrackingState() == ORB_SLAM2::Tracking::OK) {
                {
                    cv::Mat track_img;
                    drawKeypoints(subscribed_image, SLAM.GetTrackedKeyPointsUn(), track_img, cv::Scalar(0,255,0));
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
                            pcl::PointXYZINormal p;
                            p.x = static_cast<float>(point.at<float>(0));
                            p.y = static_cast<float>(point.at<float>(1));
                            p.z = static_cast<float>(point.at<float>(2));
                            float weight = 1.0f;
                            p.intensity = weight;
                            vslam_data->push_back(p);
                        }
                    }
                    pcl_conversions::toPCL(process_stamp, vslam_data->header.stamp);
                    vslam_data->header.frame_id = "world";
                    vslam_publisher.publish(vslam_data);
                }
            }

            // Inform processing time
            std::stringstream ss;
            long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count();
            ss << "processing time= \033[35m"
                << time_ms
                << "\033[m ms";
            ROS_INFO("%s", ss.str().c_str());
        }
        // Reset input
        subscribed_image = cv::Mat();
        // Spin and wait
        loop_rate.sleep();
        ros::spinOnce();
    }
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

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


