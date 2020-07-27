#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;




std::vector<String> getOutputsNames(const Net& net);
void postprocess(Mat& frame, const std::vector<Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;

std::string classesFile = "coco.names";
std::ifstream ifs(classesFile.c_str());

Net net = readNetFromDarknet("yolo.cfg", "yolo.weights");


std::vector<std::string> classes;

int main(int argc, char* argv[])
{
	std::string line;
	while (ifs){
		 std::getline(ifs, line);
		 classes.push_back(line);
	}
	VideoCapture cap(0);
	VideoWriter video;
	video.open("test.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	Mat frame;
	Mat blob;
	while (true) {
		cap >> frame;
		blobFromImage(frame, blob, 1 / 2255.f, Size(416, 416), Scalar(0,0,0), true, false);
		net.setInput(blob);
		std::vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));
		postprocess(frame, outs);
		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		video.write(detectedFrame);
		imshow("img", detectedFrame);
		if (waitKey(10) == 27) {
			break;
		}
	}
	return 0;
}

std::vector<String> getOutputsNames(const Net& net) {
	static std::vector<String> names;
	if (names.empty()) {
		std::vector<int> outLayers = net.getUnconnectedOutLayers();
		std::vector<String> layersNames = net.getLayerNames();
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i) {
			names[i] = layersNames[outLayers[i] - 1];
		}
	}
	return names;
}

void postprocess(Mat& frame, const std::vector<Mat>& outs) {
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i) {
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold) {
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}
	std::vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i) {
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

	std::string label = format("%.2f", conf);
	if (!classes.empty()) {
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}