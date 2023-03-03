#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/face.hpp"
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::face;
void detectAndDisplay(Mat frame, Ptr<FaceRecognizer> model);
bool check_file_existence(const string& name);
CascadeClassifier face_cascade;
string face_cascade_file_path = "F:/OpenCV_no_contrib/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
string working_dir = "E:/faces/";
void trainFaces(string dir) {
    Ptr<EigenFaceRecognizer> model_train = EigenFaceRecognizer::create();
    vector<Mat> images;
    vector<int> labels;
    Mat image_gray;
    for (int i = 0; i < 100; i++)
        for (int j = 1; j <= 10; j++) {
            string image_path = dir + to_string(i) + to_string(j) + ".jpg";
            if (check_file_existence(image_path)) {
                Mat image = imread(image_path, 0);
                //cvtColor(image, image_gray, COLOR_BGR2GRAY);
                //equalizeHist(image_gray, image_gray);              
                images.push_back(image);
                labels.push_back(i);
            }
        }
    model_train->train(images, labels);
    model_train->save(dir + "trained_model.yml");
}
bool check_file_existence(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}
void addFace(string dir, int label) {
    VideoCapture capture;
    capture.open(0);
    Mat frame, frame_gray;
    int i = 1;
    while (capture.read(frame)) {
        if (!frame.empty()) {
            cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
            equalizeHist(frame_gray, frame_gray);
            vector<Rect> faces;
            face_cascade.detectMultiScale(frame_gray, faces);
            Mat face = frame(faces[0]);
            Mat face_resized;
            resize(face, face_resized, Size(520, 520), 1.0, 1.0, INTER_CUBIC);
            imwrite(dir + to_string(label) + to_string(i) + ".jpg", face_resized);
            i++;
            if (i == 10)
                break;
            Point x1(faces[0].x, faces[0].y);
            Point x2(faces[0].x + faces[0].height, faces[0].y + faces[0].width);
            rectangle(frame, x1, x2, Scalar(0, 255, 0), 2);            
            imshow("Face Addition Frame", frame);
        }
        if (waitKey(10) == 27)
        {
            break;
        }
    }
}
int main(int argc, const char** argv)
{    
    if (!face_cascade.load(face_cascade_file_path))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    //addFace(working_dir, 1);
    trainFaces(working_dir);
    int camera_device = 0;
    VideoCapture capture;
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->read("E:/faces/trained_model.yml");
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        detectAndDisplay(frame, model);
        if (waitKey(10) == 27)
        {
            break;
        }
    }    
    return 0;
}

void detectAndDisplay(Mat frame, Ptr<FaceRecognizer> model)
{
    Mat frame_gray;   
    Mat frame_original = frame.clone();
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    
    for (int i = 0; i < faces.size(); i++)
    {
        Mat face = frame_gray(faces[i]);
        Mat face_resized;
        resize(face, face_resized, Size(520, 520), 1.0, 1.0, INTER_CUBIC);
        Point x1(faces[i].x, faces[i].y);
        Point x2(faces[i].x + faces[i].height, faces[i].y + faces[i].width);
        rectangle(frame_original, x1, x2, Scalar(0, 255, 0), 2);
        int label = -1; double confidence = 0;
        model->predict(face_resized, label, confidence);
        cout << label << " " << confidence << endl;
        int pos_x = max(faces[i].tl().x - 10, 0);
        int pos_y = max(faces[i].tl().y - 10, 0);
        putText(frame_original, to_string(label), Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);
    }
    imshow("Live Face Detection", frame_original);
}
