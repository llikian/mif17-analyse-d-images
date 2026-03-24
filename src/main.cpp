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

#include "Vibe.h"

static void mergeAndRefineModels(const Mat& mask1, Mat& mask2, Mat& out)
{
    bitwise_and(mask1, mask2, out);

    int niters = 2;

    dilate(out, out, Mat(), Point(-1,-1), niters);
    erode(out, out, Mat(), Point(-1,-1), niters);

    // vector<vector<Point> > contours;
    // vector<Vec4i> hierarchy;
    // Mat temp;
    //
    // dilate(out, temp, Mat(), Point(-1,-1), niters);
    // erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    // dilate(temp, temp, Mat(), Point(-1,-1), niters);

    // findContours( temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    // out = Mat::zeros(mask1.size(), CV_8UC3);

    // if(contours.size() == 0)
    //     return;
    // // iterate through all the top-level contours,
    // // draw each connected component with its own random color
    // int idx = 0, largestComp = 0;
    // double maxArea = 0;
    // for( ; idx >= 0; idx = hierarchy[idx][0] )
    // {
    //     const vector<Point>& c = contours[idx];
    //     double area = fabs(contourArea(Mat(c)));
    //     if( area > maxArea )
    //     {
    //         maxArea = area;
    //         largestComp = idx;
    //     }
    // }
    // Scalar color(255, 255, 255);
    // drawContours( out, contours, largestComp, color, FILLED, LINE_8, hierarchy );
}


int main() {
    try {
        VideoCapture video = VideoCapture("data/Webcam.mp4");
        Mat tmp_frame, frame_ycrcb, frame_gray,out_vibe, bgmask, out_mog, out_final;

        ViBe vibe;
        bool count = true;

        Ptr<BackgroundSubtractorMOG2> MOG2Substractor = createBackgroundSubtractorMOG2();
        double alpha = 0.005; // MOG2 learning rate - lower = slower but more stable
        MOG2Substractor->setVarThreshold(20); // Background threshold
        MOG2Substractor->setDetectShadows(false);
        MOG2Substractor->setHistory(1000);

        namedWindow("video", 1);
        namedWindow("ViBe", 1);
        namedWindow("MOG2", 1);
        namedWindow("final", 1);

        while (video.isOpened()) {
            if (!video.read(tmp_frame)) {
                std::cout << "Can't receive frame (stream end?). Exiting ..." << std::endl;
                break;
            }

            //// Current frame processing

            imshow("video", tmp_frame);

            // MOG2 Processing
            // YCrCb is light independent and other channels change less than HSV
            cvtColor(tmp_frame, frame_ycrcb, COLOR_BGR2YCrCb);

            MOG2Substractor->apply(tmp_frame, out_mog, alpha);
            // MOG2Substractor->apply(tmp_frame, bgmask, alpha);
            // refineSegments(tmp_frame, bgmask, out_mog);

            imshow("MOG2", out_mog);

            //ViBe Processing
            cvtColor(tmp_frame, frame_gray, COLOR_RGB2GRAY);
            if (count)
            {
                vibe.init(frame_gray);
                vibe.ProcessFirstFrame(frame_gray);
                cout<<"Training ViBe Success."<<endl;
                count=false;
            }
            else
            {
                vibe.Run(frame_gray);
                out_vibe = vibe.getFGModel();
                // morphologyEx(FGModel, FGModel, MORPH_OPEN, Mat());
                imshow("ViBe", out_vibe);

                mergeAndRefineModels(out_mog, out_vibe, out_final);
                imshow("final", out_final);
            }

            //// End of Current frame processing

            char keycode = (char)waitKey(30);
            if(keycode == 27)
                break;
        }

    } catch(const std::exception& exception) {
        std::cerr << "ERROR : " << exception.what() << '\n';
        return -1;
    }

    return 0;
}
