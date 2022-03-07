#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include "video-analysis/face.hpp"

using namespace std;
using namespace cv;

const string WindowName = "Face Detection example";

int faceRegWithDnn() {
    shared_ptr<pf::FaceRecognizer> faceRecoginzer = make_shared<pf::FaceRecognizer>("./data/face_recognition_sface_2021dec.onnx", 
                                                                                    "./data/face_detection_yunet_2021dec.onnx");

    VideoCapture VideoStream(0);
    if (!VideoStream.isOpened())
    {
        printf("Error: Cannot open video stream from camera\n");
        return 1;
    }

    Mat referenceFrame;
    do
    {
        VideoStream >> referenceFrame;
        faceRecoginzer->setDetectorSize(referenceFrame.cols, referenceFrame.rows);
        auto faces = faceRecoginzer->detect(referenceFrame);

        for (size_t i = 0; i < faces.rows; i++)
        {
            int x = (int) faces.row(i).at<float>(0,0);
            int y = (int) faces.row(i).at<float>(0,1);

            int width = (int) faces.row(i).at<float>(0,2);
            int height = (int) faces.row(i).at<float>(0,3);

            float score = faces.row(i).at<float>(0,14);

            Rect face(x, y, width, height);
            rectangle(referenceFrame, face, Scalar(0,255,0));
            putText(referenceFrame, to_string(score), Point(x+width, y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200,200,250));
            cout << x << "," << y << "," << width << "," << height << endl;
        }
        
        imshow(WindowName, referenceFrame);
        
    } while (waitKey(30) < 0);

    return 0;
}

void showImageTest(String filename) {
    Mat imgMat;
    imgMat = imread(filename, IMREAD_COLOR);
    imshow(WindowName, imgMat);
    waitKey();
}

int main()
{
    faceRegWithDnn();
    return 0;
}