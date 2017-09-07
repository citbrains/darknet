
#ifndef __CIT_YOLO_H__
#define __CIT_YOLO_H__

#include <vector>
#include <opencv2/opencv.hpp>

enum LABEL
{
	LABEL_BALL,
	LABEL_GOALPOST,
	LABELNUM
};

struct yolo_predict_data
{
	float prob;
	float x;
	float y;
	float w;
	float h;
};

extern void yolo_init(char *, char *, int *);
extern void yolo_fina(void);
extern void yolo_predict(IplImage *);
extern void yolo_get_object(int, int, int &, int &, int &, int &, float &);
extern void yolo_get_bounding_boxes(std::vector<struct yolo_predict_data> &, int);

#endif // __CIT_YOLO_H__

