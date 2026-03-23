/***************************************************************************************************
 * @file  main.cpp
 * @brief Contains the main program of the project
 **************************************************************************************************/

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video/background_segm.hpp>

using namespace cv;
using namespace std;

static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
    int niters = 3;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat temp;
    dilate(mask, temp, Mat(), Point(-1,-1), niters);
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    dilate(temp, temp, Mat(), Point(-1,-1), niters);
    findContours( temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    dst = Mat::zeros(img.size(), CV_8UC3);
    if( contours.size() == 0 )
        return;
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0, largestComp = 0;
    double maxArea = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;
        }
    }
    Scalar color(0, 0, 255);
    drawContours( dst, contours, largestComp, color, FILLED, LINE_8, hierarchy );
}

int main() {
    try {
        VideoCapture video = VideoCapture("data/Webcam.mp4");
        Mat tmp_frame, bgmask, out_frame;
        bool update_bg_model = true;

        namedWindow("video", 1);
        namedWindow("segmented", 1);

        Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
        bgsubtractor->setVarThreshold(10);

        while (video.isOpened()) {
            if (!video.read(tmp_frame)) {
                std::cout << "Can't receive frame (stream end?). Exiting ..." << std::endl;
                break;
            }

            // Current frame processing

            bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);
            refineSegments(tmp_frame, bgmask, out_frame);

            // End of Current frame processing

            imshow("video", tmp_frame);
            imshow("segmented", out_frame);

            char keycode = (char)waitKey(30);
            if(keycode == 27)
                break;

            if(keycode == ' ')
            {
                update_bg_model = !update_bg_model;
                printf("Learn background is in state = %d\n",update_bg_model);
            }
        }

    } catch(const std::exception& exception) {
        std::cerr << "ERROR : " << exception.what() << '\n';
        return -1;
    }

    return 0;
}
