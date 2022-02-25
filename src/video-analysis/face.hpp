// This file is part of Project-F
// TODO(Nengsheng Qian): add descriptions

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include "../core/base64.hpp"

using namespace std;
using namespace cv;

namespace pf
{

const int DEFAUTL_INPUT_W = 320;
const int DEFAUTL_INPUT_H = 320;

const float NMS_THRESHOLD = 0.6f;

class FaceRecognizer
{
public:
    FaceRecognizer(string recognizerModel, string detectorModel)
    {
        _recognizerNet = FaceRecognizerSF::create(recognizerModel, "", 0, 0);
        _detectorNet = FaceDetectorYN::create(detectorModel,"" , cv::Size(DEFAUTL_INPUT_W, DEFAUTL_INPUT_H), 0.9f, NMS_THRESHOLD);
    }

    ~FaceRecognizer(){ }

    void setDetectorSize(int width, int height)
    {
        _detectorNet->setInputSize(cv::Size(width, height));
    }

    Mat detect(Mat image)
    {
        Mat faces;
        _detectorNet->detect(image, faces);

        return faces;
    }

    string feature(Mat image)
    {
        // TODO(Nengsheng Qian): compute image feature
        Mat feature;
        _recognizerNet->feature(image, feature);
        return base64_encode(feature.ptr(0), feature.cols * feature.rows);
    }
    
private:
    Ptr<FaceRecognizerSF> _recognizerNet;
    Ptr<FaceDetectorYN> _detectorNet;
    shared_ptr<Size> detector_size;
};

} // pf