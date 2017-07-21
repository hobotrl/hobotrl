/********************************************************
*   
*   Filename: images.cpp
*   Author  : H.WEI
*   Date    : 2017-06-21
*   Describe: Used for receive image from simulator 
*
********************************************************/
#include <ros/ros.h>
#include <image_transport/image_transport.h> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

int img_index = 0;

void ImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try
	{
        char path[1000] = {0};
        sprintf(path,"/media/henglaiwei/1/%d.jpg",img_index++);     
        cv::imwrite(path,cv_bridge::toCvShare(msg,"bgr8")->image);
        }

    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Coule not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

}
int main(int argc, char **argv)
{
	ros::init(argc, argv, "image_listener");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	image_transport::Subscriber sub = it.subscribe("/training/image",1,ImageCallback);
	ros::spin();
}
