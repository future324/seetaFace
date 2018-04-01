#pragma once
#include<opencv2\opencv.hpp>
#include<Seetaface\FaceDetection\face_detection.h>
#include<Seetaface\FaceAlignment\face_alignment.h>
#include<Seetaface\FaceIdentification\face_identification.h>
using namespace std;
using namespace cv;
using namespace seeta;
int libsize = 49;
struct Identity
{
	string name;
	FaceFeatures feature;
	Mat pic;
};
struct SimilarityResult {
	int libidx;
	float simi;
};
bool simiCmp(SimilarityResult &a, SimilarityResult &b) {
	return a.simi > b.simi;
}

FaceDetection faceDetector("C:\\opencv\\include\\Seetaface\\seeta_fd_frontal_v1.0.bin");
FaceAlignment faceAlgnment("C:\\opencv\\include\\Seetaface\\seeta_fa_v1.1.bin");
FaceIdentification faceRecognizer("C:\\opencv\\include\\Seetaface\\seeta_fr_v1.0.bin");
vector<Identity> IdentitiesLib(libsize);
vector<SimilarityResult>simirslt(libsize);
class Recongnizer
{
public:
	void Init();
	void Recongnize(Mat );

private:
	ImageData fmtg(Mat &src) {
		Mat gray;
		cvtColor(src, gray, cv::COLOR_BGR2GRAY);
		seeta::ImageData img_data;
		img_data.data = gray.data;
		img_data.width = gray.cols;
		img_data.height = gray.rows;
		img_data.num_channels = gray.channels();
		return img_data;
	}
	ImageData fmt(Mat &src) {
		seeta::ImageData img_data;
		img_data.data = src.data;
		img_data.width = src.cols;
		img_data.height = src.rows;
		img_data.num_channels = src.channels();
		return img_data;
	}
	Mat fmt(ImageData &src) {
		return Mat(Size(src.width, src.width), CV_MAKETYPE(CV_8U, src.num_channels), src.data);
	}
	cv::Rect bbox2Rect(seeta::Rect &rect) {
		return cv::Rect(rect.x, rect.y, rect.width, rect.height);
	}
};
void Recongnizer::Init()
{
	faceDetector.SetMinFaceSize(40);
	faceDetector.SetScoreThresh(1.5f);
	faceDetector.SetImagePyramidScaleFactor(1.f);
	faceDetector.SetWindowStep(4, 4);

	for (int i = 0; i < libsize; i++) 
	{
		auto name = string("C:\\Users\\xieyo\\Documents\\Visual Studio 2017\\Projects\\seetaFace\\data\\src\\") + to_string(i + 1);
		IdentitiesLib[i].pic = imread(name + string(".jpg"));
		IdentitiesLib[i].name = i+1;
		auto color = fmt(IdentitiesLib[i].pic);
		auto gray = fmtg(IdentitiesLib[i].pic);
		auto detected = faceDetector.Detect(gray);
		if (detected.size())
		{
			FacialLandmark landmark[5];
			faceAlgnment.PointDetectLandmarks(gray, detected[0], landmark);
			FaceFeatures feature = new float[2048];
			faceRecognizer.ExtractFeatureWithCrop(color, landmark, feature);
			IdentitiesLib[i].feature = feature;
			IdentitiesLib[i].name = name;
			cout << ".";
		}
	}
	
}
void Recongnizer::Recongnize(Mat frame)
{
	
	Identity currentFace;
	auto gray = fmtg(frame);
	auto detected = faceDetector.Detect(gray);
	for (size_t i = 0; i < detected.size(); i++)
	{
		auto bbox = detected[i].bbox;
		auto faceRect = bbox2Rect(bbox);
		rectangle(frame, faceRect, CV_RGB(128, 128, 255),5);
		if (faceRect.x < 0)    faceRect.x = 0;
		if (faceRect.y < 0)    faceRect.y = 0;
		if ((faceRect.x + faceRect.width) >= 640)
		{
			faceRect.width = 640 - faceRect.x;
		}
		if( faceRect.y + faceRect.height >= 480 )
		{
			faceRect.height = 480 - faceRect.y;
		}
		currentFace.pic = frame(faceRect);//框还没有加位置边界处理，框出去了就崩
		FacialLandmark landmark[5];
		faceAlgnment.PointDetectLandmarks(gray, detected[i], landmark);
		FaceFeatures feature = new float[2048];
		faceRecognizer.ExtractFeatureWithCrop(fmt(frame), landmark, feature);
		for (size_t j = 0; j < libsize; j++) {
			simirslt[j].libidx = j;
			simirslt[j].simi = faceRecognizer.CalcSimilarity(feature, IdentitiesLib[j].feature);
		}
		sort(simirslt.begin(), simirslt.end(), simiCmp);
		//cout << IdentitiesLib[i].name;
	//	imshow("0", IdentitiesLib[simirslt[0].libidx].pic);
		//for (size_t j = 0; j < 2; j++)
		//{
		//	//cout << "Name:" << IdentitiesLib[simirslt[j].libidx].name << endl;
		//	//cout << "Confidence:" << simirslt[j].simi << endl;
		//	if (simirslt[j].simi>0.6)
		//	{
		//		cout << IdentitiesLib[simirslt[j].libidx].name;
		//		imshow(to_string(j), IdentitiesLib[simirslt[j].libidx].pic);
		//	}
		//}
	}
}