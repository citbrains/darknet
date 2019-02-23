#include "darknet_yolo.h"
#include <iostream>
#include <algorithm>

YOLO_Darknet::YOLO_Darknet(char *cfg, char *weight, int *threshold) : predict_ball(0), predict_goalpost(0), img_width(320), img_height(240)
{
	setGPUMode(0);
    net = load_network(cfg,weight,0);
    set_batch_network(net,1);
	resize = cvCreateImage(cvSize(416, 416), 8, 3);
	dl = net->layers[net->n-1];
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

static void ipl_into_image(IplImage *src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;

    for(int i = 0; i < h; i++){
        for(int k = 0; k < c; k++){
            for(int j = 0; j < w; j++){
                im.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 255.0;
            }
        }
    }
}

static image ipl_to_image(IplImage *src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

void YOLO_Darknet::detectObjectsUsingYOLO(IplImage *src)
{
	this->img_width = src->width;
	this->img_height = src->height;
	cvResize(src, resize);
	cvCvtColor(resize, resize, CV_YCrCb2BGR);
	image im = ipl_to_image(resize);
    image sized = letterbox_image(im, net->w, net->h);
	float *X = sized.data;
	network_predict(net, X);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, 0.0, 0.5, 0, 1, &nboxes); // hier_thresh(default 0.5)
	getProbs(dets,nboxes);
	free_image(im);
    free_image(sized);
    free_detections(dets,nboxes);
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

void YOLO_Darknet::getProbs(detection *dets, int nbox)
{
	const int num = nbox;
	predict_ball.clear();
	predict_goalpost.clear();
	for(int i = 0; i < num; i++) {
		if(dets[i].prob[LABEL_BALL] > thresh[LABEL_BALL]) {
			const struct yolo_predict_data pr = copyProbData(dets[i].prob[LABEL_BALL], dets[i].bbox);
			predict_ball.push_back(pr);
			std::sort(predict_ball.begin(), predict_ball.end(), SORT_PROB_IN_DESCENDING_ORDER());
		}
		if(dets[i].prob[LABEL_GOALPOST] > thresh[LABEL_GOALPOST]){
			const struct yolo_predict_data pr = copyProbData(dets[i].prob[LABEL_GOALPOST], dets[i].bbox);
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

int YOLO_Darknet::maxIndexVec(float *vec)
{
    int idx=-1;
    float max_elem = 0.0;
    for(int i=0;i<dl.classes;i++)
    {
        if(vec[idx] > max_elem){
            max_elem = vec[idx];
            idx = i;
        }
    }
	return idx;
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

int YOLO_Darknet::getObjectPos(int label, int index, int &w, int &h, int &x, int &y, float &score)
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
		return 0;
	} else {
		score = 0.0;
		x = -1;
		y = -1;
		w = -1;
		h = -1;
		return 1;
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

void YOLO_Darknet::setThreshold(int labelnum, int threshold)
{
    this->thresh[labelnum] = (float)threshold / 100.0;
}
