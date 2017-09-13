#include "darknet_yolo.h"
#include <iostream>
#include <algorithm>

YOLO_Darknet::YOLO_Darknet(char *cfg, char *weight, int *threshold) : predict_ball(0), predict_goalpost(0), img_width(320), img_height(240)
{
	setGPUMode(0);
	net = parse_network_cfg(cfg);
	if(weight) {
		load_weights(&net, weight);
	}
	resize = cvCreateImage(cvSize(448, 448), 8, 3);
	dl = net.layers[net.n-1];
	boxes.resize(dl.side * dl.side * dl.n);
	probs.resize(dl.side * dl.side * dl.n);
	for(int i = 0; i < dl.side * dl.side * dl.n; i++)
		probs[i].resize(dl.classes);
	thresh.resize(LABELNUM);
	for(int i = 0; i < LABELNUM; i++)
		thresh[i] = (float)threshold[i] / 100.0;
}

YOLO_Darknet::~YOLO_Darknet()
{
}

void YOLO_Darknet::detectObjectsUsingYOLO(IplImage *src)
{
	this->img_width = src->width;
	this->img_height = src->height;
	set_batch_network(&net, 1);
	cvResize(src, resize);
	cvCvtColor(resize, resize, CV_YCrCb2BGR);
	image im = ipl_to_image(resize);
	float *X = im.data;
	float *predictions = network_predict(net, X);
	convert_detection(predictions, 1, 1, 0);
	getProbs();
	free_image(im);
}

void YOLO_Darknet::convert_detection(float *predictions, int w, int h, int only_objectness)
{
	const int classes = dl.classes;
	const int num = dl.n;
	const int square = dl.sqrt;
	const int side = dl.side;
	for(int i = 0; i < side * side; i++) {
		const int row = i / side;
		const int col = i % side;
		for(int n = 0; n < num; n++) {
			const int index = i * num + n;
			const int p_index = side * side * classes + i * num + n;
			const float scale = predictions[p_index];
			const int box_index = side * side * (classes + num) + (i * num + n) * 4;
			boxes[index].x = (predictions[box_index + 0] + col) / side * w;
			boxes[index].y = (predictions[box_index + 1] + row) / side * h;
			boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
			boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
			for(int j = 0; j < classes; j++) {
				const int class_index = i * classes;
				const float prob = scale * predictions[class_index+j];
				probs[index][j] = prob;
			}
			if(only_objectness) {
				probs[index][0] = scale;
			}
		}
	}
}

struct yolo_predict_data YOLO_Darknet::copyProbData(float prob, box b)
{
	struct yolo_predict_data d;
	d.prob = prob;
	d.x = b.x;
	d.y = b.y;
	d.w = b.w;
	d.h = b.h;
	return d;
}

void YOLO_Darknet::getProbs(void)
{
	const int num = dl.side * dl.side * dl.n;
	predict_ball.clear();
	predict_goalpost.clear();
	for(int i = 0; i < num; i++) {
		const int cls = maxIndexVec(probs[i]);
		const float prob = probs[i][cls];
		if((prob > thresh[LABEL_BALL]) && (cls == LABEL_BALL)) {
			const struct yolo_predict_data pr = copyProbData(prob, boxes[i]);
			predict_ball.push_back(pr);
			std::sort(predict_ball.begin(), predict_ball.end(), SORT_PROB_IN_DESCENDING_ORDER());
		}
		if((prob > thresh[LABEL_GOALPOST]) && (cls == LABEL_GOALPOST)){
			const struct yolo_predict_data pr = copyProbData(prob, boxes[i]);
			bool ret = false;
			for(int n = 0; n < predict_goalpost.size(); n++) {
				ret |= isOverlap(pr, predict_goalpost[n]);
			}
			if(!ret) {
				predict_goalpost.push_back(pr);
				std::sort(predict_goalpost.begin(), predict_goalpost.end(), SORT_PROB_IN_DESCENDING_ORDER());
			}
		}
	}
}

void YOLO_Darknet::setGPUMode(int gpu_index)
{
#ifdef GPU
	if(gpu_index >= 0) {
		cuda_set_device(gpu_index);
	}
#endif
}

int YOLO_Darknet::maxIndexVec(std::vector<float> &vec)
{
	if(vec.size() == 0) return 0;
	return std::max_element(vec.begin(), vec.end()) - vec.begin();
}

bool YOLO_Darknet::isOverlap(struct yolo_predict_data box1, struct yolo_predict_data box2)
{
	const float X1 = std::max(box1.x-(box1.w/2), box2.x-(box2.w/2));
	const float Y1 = std::max(box1.y-(box1.h/2), box2.y-(box2.h/2));
	const float X2 = std::min(box1.x+(box1.w/2), box2.x+(box2.w/2));
	const float Y2 = std::min(box1.y+(box1.h/2), box2.y+(box2.h/2));
	if ((X1 < X2) && (Y1 < Y2)) {
		return true;
	} else {
		return false;
	}
}

void YOLO_Darknet::getObjectPos(int label, int index, int &w, int &h, int &x, int &y, float &score)
{
	std::vector<struct yolo_predict_data> obj = predict_ball;
	if(label == LABEL_BALL) {
		obj = predict_ball;
	} else if(label == LABEL_GOALPOST) {
		obj = predict_goalpost;
	}
	if(!obj.empty() && index < obj.size()) {
		score = obj[index].prob;
		x = (int)(obj[index].x * img_width);
		y = (int)(obj[index].y * img_height);
		w = (int)(obj[index].w * img_width);
		h = (int)(obj[index].h * img_height);
	} else {
		score = 0.0;
		x = -1;
		y = -1;
		w = -1;
		h = -1;
	}
}

void YOLO_Darknet::getBoundingBoxes(std::vector<struct yolo_predict_data> &data, int label)
{
	if(label == LABEL_BALL) {
		for(auto p: predict_ball)
			data.push_back(p);
	} else if(label == LABEL_GOALPOST) {
		for(auto p: predict_goalpost)
			data.push_back(p);
	} else {
		data.clear();
	}
	return;
}

