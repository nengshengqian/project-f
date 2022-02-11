#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include "video-analysis/face-detect.cpp"

using namespace std;
using namespace cv;

const string WindowName = "Face Detection example";

int face_detect(String videoAddress) {
    namedWindow(WindowName);

    VideoCapture VideoStream(videoAddress);

    if (!VideoStream.isOpened())
    {
        printf("Error: Cannot open video stream from camera\n");
        return 1;
    }

    std::string cascadeFrontalfilename = samples::findFile("./data/lbpcascade_frontalface.xml");
    cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);
    if ( cascade->empty() )
    {
      printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
      return 2;
    }

    cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);
    if ( cascade->empty() )
    {
      printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
      return 2;
    }

    DetectionBasedTracker::Parameters params;
    DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

    if (!Detector.run())
    {
        printf("Error: Detector initialization failed\n");
        return 2;
    }

    Mat ReferenceFrame;
    Mat GrayFrame;
    vector<Rect> Faces;

    do
    {
        VideoStream >> ReferenceFrame;
        cvtColor(ReferenceFrame, GrayFrame, COLOR_BGR2GRAY);
        Detector.process(GrayFrame);
        Detector.getObjects(Faces);

        for (size_t i = 0; i < Faces.size(); i++)
        {
            rectangle(ReferenceFrame, Faces[i], Scalar(0,255,0));
        }

        imshow(WindowName, ReferenceFrame);
    } while (waitKey(30) < 0);

    Detector.stop();
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
    return face_detect("rtsp://ssss:554");
}