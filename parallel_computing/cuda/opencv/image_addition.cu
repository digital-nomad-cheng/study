#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat h_img1 = cv::imread("images/cameraman.tif");
    cv::Mat h_img2 = cv::imread("images/circles.png");
    cv::Mat h_result;

    // Create Memory for storing images on device
    cv::cuda::GpuMat d_result1, d_img1, d_img2;
    d_img1.upload(h_img1);
    d_img2.upload(h_img2);

    cv::cuda::add(d_img1, d_img2, d_result1);
    d_result1.download(h_result1);
    cv::imshow("result", h_result1);
    cv::waitKey(0);

    return 0;
}
