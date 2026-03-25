/***************************************************************************************************
 * @file  main.cpp
 * @brief Contains the main program of the project
 **************************************************************************************************/

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "Vibe.h"

static void mergeAndRefineModels(const cv::Mat& mask1, cv::Mat& mask2, cv::Mat& out) {
    bitwise_and(mask1, mask2, out);

    int niters = 2;

    dilate(out, out, cv::Mat(), cv::Point(-1, -1), niters);
    erode(out, out, cv::Mat(), cv::Point(-1, -1), niters);

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

auto ensureColor = [](cv::Mat& img) {
    if (img.channels() == 1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
};

void addLabel(cv::Mat& img, const std::string& text)
{
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.7;
    int thickness = 2;

    // shadow
    cv::putText(img, text, cv::Point(10, 30), font, scale, cv::Scalar(255,0,127), thickness);
}

int main() {
    try {
        cv::VideoCapture video = cv::VideoCapture(0);
        cv::Mat tmp_frame, frame_ycrcb, frame_gray, out_vibe, bgmask, out_mog, img_top, img_bot, out_final;

        ViBe vibe;
        bool count = true;

        cv::Ptr<cv::BackgroundSubtractorMOG2> MOG2Substractor = cv::createBackgroundSubtractorMOG2();
        double alpha = 0.005;                 // MOG2 learning rate - lower = slower but more stable
        MOG2Substractor->setVarThreshold(20); // Background threshold
        MOG2Substractor->setDetectShadows(false);
        MOG2Substractor->setHistory(1000);

        cv::namedWindow("video", 1);

        while(video.isOpened()) {
            if(!video.read(tmp_frame)) {
                std::cout << "Can't receive frame (stream end?). Exiting ...\n";
                break;
            }

            //// Current frame processing

            // MOG2 Processing
            // YCrCb is light independent and other channels change less than HSV
            cv::cvtColor(tmp_frame, frame_ycrcb, cv::COLOR_BGR2YCrCb);

            MOG2Substractor->apply(tmp_frame, out_mog, alpha);

            // ViBe Processing
            cv::cvtColor(tmp_frame, frame_gray, cv::COLOR_RGB2GRAY);
            if(count) {
                vibe.init(frame_gray);
                vibe.ProcessFirstFrame(frame_gray);
                std::cout << "Training ViBe Success." << std::endl;
                count = false;
            } else {
                vibe.Run(frame_gray);
                out_vibe = vibe.getFGModel();
                // morphologyEx(FGModel, FGModel, MORPH_OPEN, Mat());

                mergeAndRefineModels(out_mog, out_vibe, out_final);

                cv::resize(out_final, out_final, tmp_frame.size());
                cv::resize(out_mog, out_mog, tmp_frame.size());
                cv::resize(out_vibe, out_vibe, tmp_frame.size());

                ensureColor(out_final);
                ensureColor(out_mog);
                ensureColor(out_vibe);

                addLabel(tmp_frame, "Video");
                addLabel(out_final, "Final");
                addLabel(out_mog, "MOG2");
                addLabel(out_vibe, "ViBe");

                cv::hconcat(tmp_frame, out_final, img_top);
                cv::hconcat(out_mog, out_vibe, img_bot);
                cv::vconcat(img_top, img_bot, out_final);

                imshow("video", out_final);
            }

            //// End of Current frame processing

            char keycode = (char) cv::waitKey(30);
            if(keycode == 27) { break; }
        }

    } catch(const std::exception& exception) {
        std::cerr << "ERROR : " << exception.what() << '\n';
        return -1;
    }

    return 0;
}
