#pragma once

#ifdef GPU
#include "cuda.h"
#endif //GPU
#ifdef __cplusplus
//extern "C" {
#endif //__cplusplus
#include "parser.h"
#include "utils.h"
#include "blas.h"
#include "connected_layer.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "box.h"
#include "demo.h"
#include "image.h"
#include "network.h"
#ifdef __cplusplus
//}
#endif //__cplusplus
#include <opencv2/opencv.hpp>
#include <vector>

#include "cit_yolo.h"

extern float *network_predict(network, float *);

class YOLO_Darknet
{
public:
	struct SORT_PROB_IN_DESCENDING_ORDER
	{
		inline bool operator() (const struct yolo_predict_data &lhs, const struct yolo_predict_data &rhs)
		{
			return (lhs.prob > rhs.prob);
		}
	};
	std::vector<struct yolo_predict_data> predict_ball;
	std::vector<struct yolo_predict_data> predict_goalpost;
	YOLO_Darknet(char *, char *, int *);
	~YOLO_Darknet();
	void detectObjectsUsingYOLO(IplImage *);
	void getObjectPos(int, int, int &, int &, int &, int &, float &);
	std::vector<struct yolo_predict_data> getBoundingBoxes(int);

private:
	std::vector<float> thresh;
	std::vector< std::vector<float> > probs;
	std::vector<box> boxes;
	detection_layer dl;
	network net;
	IplImage *resize;
	int img_width, img_height;
	float nms;
	void setGPUMode(int);
	void convert_detection(float *, int, int, int);
	void getProbs(void);
	bool isOverlap(struct yolo_predict_data &, struct yolo_predict_data &);
	int maxIndexVec(std::vector<float> &);
	struct yolo_predict_data copyProbData(float, box);
};

