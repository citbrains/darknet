
#include "darknet_yolo.h"
#include "cit_yolo.h"

static YOLO_Darknet *yolo;

void yolo_init(char *cfg, char *weight, int *thresholds)
{
	yolo = new YOLO_Darknet(cfg, weight, thresholds);
	return;
}

void yolo_fina(void)
{
	delete yolo;
	return;
}

void yolo_predict(IplImage *img)
{
	yolo->detectObjectsUsingYOLO(img);
	return;
}

void yolo_get_object(int label, int index, int &weight, int &height, int &x, int &y, float &score)
{
	yolo->getObjectPos(label, index, weight, height, x, y, score);
	return;
}

void yolo_get_bounding_boxes(std::vector<struct yolo_predict_data> &boxes, int object_type)
{
	boxes = yolo->getBoundingBoxes(object_type);
	return;
}

