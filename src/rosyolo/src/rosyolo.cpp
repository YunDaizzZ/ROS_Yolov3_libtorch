#include <ros/ros.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

#define IMG_SIZE 416
#define NUM_CLASSES 20
#define CONF_THRES 0.5
#define NMS_THRES 0.3

int anchors[3][6] = {116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119, 10, 13, 16, 30, 33, 23};
string class_names[20] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

//图片预处理
static Mat letterbox_image(Mat frame)
{
    float iw, ih, w, h;
    int nw, nh;
    float scale;
    iw = frame.size().width;
    ih = frame.size().height;
    w = IMG_SIZE;
    h = IMG_SIZE;
    scale = min(w / iw, h / ih);
    nw = int(iw * scale);
    nh = int(ih * scale);

    Mat new_image(IMG_SIZE, IMG_SIZE, CV_8UC3, Scalar(128, 128, 128));
    Mat imageROI, image;

    resize(frame, image, Size(nw, nh));
    cvtColor(image, image, CV_BGR2RGB);
    imageROI = new_image(Range(floor((h - nh) / 2), floor((h - nh) / 2) + nh), Range(floor((w - nw) / 2), floor((w - nw) / 2) + nw));
    image.copyTo(imageROI);

    return new_image;
}

//去灰条
static void yolo_correct_boxes(vector<float> top, vector<float> left, vector<float> bottom, vector<float> right, vector<float> img_shape, vector<float> &boxes)
{
    float nw, nh, offsetw, offseth, scalew, scaleh;
    nw = img_shape[0] * min(IMG_SIZE / img_shape[0], IMG_SIZE / img_shape[1]);
    nh = img_shape[1] * min(IMG_SIZE / img_shape[0], IMG_SIZE / img_shape[1]);
    offsetw = (IMG_SIZE - nw) / 2. /IMG_SIZE;
    offseth = (IMG_SIZE - nh) / 2. /IMG_SIZE;
    scalew = IMG_SIZE / nw;
    scaleh = IMG_SIZE / nh;

    vector<float> box_yx, box_hw, box_mins, box_maxes;
    for (vector<Point2f>::size_type i = 0; i < top.size(); ++i)
    {
        box_yx.push_back((((top[i] + bottom[i]) / 2 / IMG_SIZE) - offseth) * scaleh);
        box_yx.push_back((((left[i] + right[i]) / 2 / IMG_SIZE) - offsetw) * scalew);
        box_hw.push_back(scaleh * (bottom[i] - top[i]) / IMG_SIZE);
        box_hw.push_back(scalew * (right[i] - left[i]) / IMG_SIZE);
    }
    for (vector<Point2f>::size_type i = 0; i < box_yx.size() / 2; ++i)
    {
        boxes.push_back(img_shape[1] * (box_yx[2 * i] - box_hw[2 * i] / 2.));
        boxes.push_back(img_shape[0] * (box_yx[2 * i + 1] - box_hw[2 * i + 1] / 2.));
        boxes.push_back(img_shape[1] * (box_yx[2 * i] + box_hw[2 * i] / 2.));
        boxes.push_back(img_shape[0] * (box_yx[2 * i + 1] + box_hw[2 * i + 1] / 2.));
    }
}

//解码
static void DecodeBox(at::Tensor input, int anchor[], at::Tensor &output)
{
    int batch_size = input.sizes()[0];
    int input_height = input.sizes()[2];
    int input_width = input.sizes()[3];

    //计算步长，每一个特征点对应原来图片上多少个像素点
    int stride_h = IMG_SIZE / input_height;
    int stride_w = IMG_SIZE / input_width;

    //归一化到特征层上，把先验框的尺寸调整到特征层大小的形式，计算出先验框在特征层上对应的宽高
    float scaled_anchors[6];
    for (int i = 0; i < 3; ++i)
    {
        scaled_anchors[2 * i] = (float)anchor[2 * i] / (float)stride_w;
        scaled_anchors[2 * i + 1] = (float)anchor[2 * i + 1] / (float)stride_h;
    }

    //对预测结果进行resize，(bs,75,13,13->bs,3,13,13,25)
    at::Tensor prediction = input.reshape({batch_size, 3, NUM_CLASSES+5, input_height, input_width}).permute({0, 1, 3, 4, 2}).contiguous();

    //先验框的中心位置、宽高调整参数
    at::Tensor x = prediction.select(4,0).sigmoid();
    at::Tensor y = prediction.select(4,1).sigmoid();
    at::Tensor w = prediction.select(4,2);
    at::Tensor h = prediction.select(4,3);

    //获得置信度，是否有物体
    at::Tensor conf = prediction.select(4,4).sigmoid();

    //获得种类置信度
    vector<long int> idx;
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        idx.push_back(i + 5);
    }
    at::Tensor indices = at::tensor(idx).to(torch::kCUDA);
    at::Tensor pred_cls = index_select(prediction, 4, indices).sigmoid();

    //生成网格，先验框中心为网格左上角，bs,3,13,13
    at::Tensor ax = at::linspace(0, input_width - 1, input_width);
    at::Tensor gx = at::rand({batch_size, 3, input_width, input_width});
    at::Tensor ay = at::linspace(0, input_height - 1, input_height);
    at::Tensor gy = at::rand({batch_size, 3, input_height, input_height});
    at::Tensor grid_x = ax.expand_as(gx).to(torch::kCUDA);
    at::Tensor grid_y = ay.expand_as(gy).transpose(3,2).to(torch::kCUDA);

    //生成先验框的宽高
    vector<float> scaled_anchors_w, scaled_anchors_h;
    at::Tensor wh = at::rand({batch_size, 1, input_height, input_height});
    for (int i = 0; i < 3; ++i)
    {
        scaled_anchors_w.push_back(scaled_anchors[2 * i]);
        scaled_anchors_h.push_back(scaled_anchors[2 * i + 1]);
    }  
    at::Tensor saw1 = at::tensor(scaled_anchors_w[0]).expand_as(wh);
    at::Tensor saw2 = at::tensor(scaled_anchors_w[1]).expand_as(wh);
    at::Tensor saw3 = at::tensor(scaled_anchors_w[2]).expand_as(wh);
    at::Tensor sah1 = at::tensor(scaled_anchors_h[0]).expand_as(wh);
    at::Tensor sah2 = at::tensor(scaled_anchors_h[1]).expand_as(wh);
    at::Tensor sah3 = at::tensor(scaled_anchors_h[2]).expand_as(wh);
    at::Tensor anchor_w = at::cat({saw1, saw2, saw3}, 1).to(torch::kCUDA);
    at::Tensor anchor_h = at::cat({sah1, sah2, sah3}, 1).to(torch::kCUDA);

    //计算调整后的先验框中心和宽高，并调整到416,416大小
    at::Tensor pred_boxes = at::rand({batch_size, 3, input_height, input_height, 4});
    pred_boxes.select(4, 0) = (x + grid_x) * stride_w;
    pred_boxes.select(4, 1) = (y + grid_y) * stride_h;
    pred_boxes.select(4, 2) = exp(w) * anchor_w * stride_w;
    pred_boxes.select(4, 3) = exp(h) * anchor_h * stride_h;

    //调整输出
    output = at::cat({pred_boxes.contiguous().view({batch_size, -1, 4}).to(torch::kCUDA), conf.contiguous().view({batch_size, -1, 1}), pred_cls.contiguous().view({batch_size, -1, NUM_CLASSES})}, 2);
}

//计算IOU
static void bbox_iou(at::Tensor box1, at::Tensor box2, at::Tensor &ious)
{
    at::Tensor b1_x1 = box1.select(1, 0).unsqueeze(0);
    at::Tensor b1_y1 = box1.select(1, 1).unsqueeze(0);
    at::Tensor b1_x2 = box1.select(1, 2).unsqueeze(0);
    at::Tensor b1_y2 = box1.select(1, 3).unsqueeze(0);
    at::Tensor b2_x1 = box2.select(1, 0).unsqueeze(0);
    at::Tensor b2_y1 = box2.select(1, 1).unsqueeze(0);
    at::Tensor b2_x2 = box2.select(1, 2).unsqueeze(0);
    at::Tensor b2_y2 = box2.select(1, 3).unsqueeze(0);

    at::Tensor inter_rect_x1 = at::max(b1_x1, b2_x1);
    at::Tensor inter_rect_y1 = at::max(b1_y1, b2_y1);
    at::Tensor inter_rect_x2 = at::min(b1_x2, b2_x2);
    at::Tensor inter_rect_y2 = at::min(b1_y2, b2_y2);
    
    at::Tensor inter_area = (inter_rect_x2 - inter_rect_x1 + 1).clamp(0, 1e+10) * (inter_rect_y2 - inter_rect_y1 + 1).clamp(0, 1e+10);
    at::Tensor b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
    at::Tensor b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

    ious = inter_area / (b1_area + b2_area - inter_area + 1e-16);
}

//非极大抑制
static void non_max_suppression(at::Tensor prediction, at::Tensor &batch_detections, int &flag)
{
    flag = 0;//用于异常判断
    //求左上角和右下角
    at::Tensor box_corner = at::zeros(prediction.sizes()).to(torch::kCUDA);
    box_corner.select(2, 0) =  prediction.select(2, 0) - prediction.select(2, 2) / 2;
    box_corner.select(2, 1) =  prediction.select(2, 1) - prediction.select(2, 3) / 2;
    box_corner.select(2, 2) =  prediction.select(2, 0) + prediction.select(2, 2) / 2;
    box_corner.select(2, 3) =  prediction.select(2, 1) + prediction.select(2, 3) / 2;

    for (int i = 0; i < 4; ++i)
    {
        prediction.select(2, i) = box_corner.select(2, i);
    }
    vector<int> output(prediction.sizes()[0], 0);

    for (int i = 0; i < prediction.sizes()[0]; ++i)
    {
        at::Tensor image_pred = prediction.select(0, i);
        //利用置信度进行第一轮筛选
        at::Tensor loc = (at::nonzero(image_pred.select(1, 4) >= 0.5)).squeeze();
        image_pred = image_pred.index_select(0, loc);

        if (image_pred.sizes()[0] == 0)
            break;
        
        //获得种类及置信度
        vector<long int> idx;
        for (int i = 0; i < NUM_CLASSES; ++i)
        {
            idx.push_back(i + 5);
        }

        at::Tensor indices = at::tensor(idx).to(torch::kCUDA);    
        tuple<at::Tensor, at::Tensor> maxcls = at::max(index_select(image_pred, 1, indices), 1);
        at::Tensor class_conf = get<0>(maxcls);
        at::Tensor class_pred = get<1>(maxcls).to(torch::kFloat);

        //获得的内容为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        at::Tensor detections = at::cat({image_pred.select(1, 0).unsqueeze(1), image_pred.select(1, 1).unsqueeze(1), image_pred.select(1, 2).unsqueeze(1), image_pred.select(1, 3).unsqueeze(1), image_pred.select(1, 4).unsqueeze(1), class_conf.unsqueeze(1), class_pred.unsqueeze(1)}, 1);
        int num_cls = 0;
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            //获得某一类初步筛选后全部的预测结果
            at::Tensor detcls = at::nonzero(detections.select(1, 6) == (float)(j + 1));
            if (detcls.sizes()[0] != 0)
            {
                flag = 1;
                //按照存在物体的置信度排序
                at::Tensor detections_class = detections.index_select(0, detcls.squeeze());
                tuple<at::Tensor, at::Tensor> sort_ret = at::sort(detections_class.select(1, 4).unsqueeze(1), 0, 1);
                detections_class = detections_class.index_select(0, get<1>(sort_ret).squeeze());

                //进行非极大抑制
                at::Tensor max_detections;
                int kk = 0;
                while (detections_class.sizes()[0])
                {
                    //取出这一类置信度最高的，一步步往下判断，判断重合度是否大于nms_thres，如果是则去除掉
                    if (kk == 0)
                    {
                        max_detections = detections_class.select(0, 0).unsqueeze(0);
                    }
                    else
                    {
                        max_detections = at::cat({max_detections, detections_class.select(0, 0).unsqueeze(0)}, 0);
                    }
                    kk++;

                    if (detections_class.sizes()[0] == 1)
                        break;

                    vector<long int> idxcls;
                    for (int k = 1; k < detections_class.sizes()[0]; ++k)
                    {
                        idxcls.push_back(k);
                    }
                    at::Tensor indices2 = at::tensor(idxcls).to(torch::kCUDA);
                    at::Tensor det_cls = index_select(detections_class, 0, indices2);
                    at::Tensor ious;
                    bbox_iou(max_detections.select(0, -1).unsqueeze(0), det_cls, ious);
                    at::Tensor loc2 = (at::nonzero(ious.squeeze() < NMS_THRES)).squeeze();

                    if (loc2.sizes()[0] != 0)
                    {
                        detections_class = det_cls.index_select(0, loc2);
                    }
                    else
                    {
                        break;
                    }
                }
                //output
                if (num_cls == 0)
                {
                    batch_detections = max_detections;
                }
                else
                {
                    batch_detections = at::cat({batch_detections, max_detections}, 0);
                }
                num_cls++;
            }
        }
    }
}

void RosYolov3()
{
    torch::jit::script::Module module = torch::jit::load("/home/bhap/Pytorch_test/YoloV3/pt/yolov3.pt");
    module.to(at::kCUDA);

    double fps = 0.0;

    VideoCapture capture(0);
    //capture.open("/home/bhap/Documents/Video/test2.mp4");
    while (1)
    {
        auto start = system_clock::now();

        Mat frame, image;

        capture >> frame;

        //图片预处理
        image = letterbox_image(frame);
        image.convertTo(image, CV_32F, 1.0 / 255);

        auto img_tensor = torch::from_blob(image.data, {1, IMG_SIZE, IMG_SIZE, 3}).permute({0, 3, 1, 2}).to(torch::kCUDA); //将Mat转化为tensor，大小为1,3,416,416
        auto img_var = torch::autograd::make_variable(img_tensor, false);  //不需要梯度

        //forward得到输出
        vector<torch::jit::IValue> inputs;
        inputs.push_back(img_var);

        auto outputs = module.forward(inputs).toTuple();
        at::Tensor output1, output2, output3, output;
        //解码
        DecodeBox(outputs->elements()[0].toTensor(), anchors[0], output1);
        DecodeBox(outputs->elements()[1].toTensor(), anchors[1], output2);
        DecodeBox(outputs->elements()[2].toTensor(), anchors[2], output3);

        output = at::cat({output1, output2, output3}, 1);
        at::Tensor batch_detections;

        //非极大抑制
        int flag;
        non_max_suppression(output, batch_detections, flag);
        if (flag)
        {
            //参数整理，tensor转array
            at::Tensor top_index = (at::nonzero(batch_detections.select(1, 4) * batch_detections.select(1, 5) > CONF_THRES)).squeeze();
            vector<int> top_label;
            vector<float> top_xmin, top_ymin, top_xmax, top_ymax, top_conf;
            for (int i = 0; i < top_index.sizes()[0]; ++i)
            {
                top_conf.push_back((batch_detections.index_select(0, top_index).select(1, 4) * batch_detections.index_select(0, top_index).select(1, 5))[i].item().toFloat());
                top_label.push_back(batch_detections.index_select(0, top_index).select(1, 6)[i].item().toInt());
                top_xmin.push_back(batch_detections.index_select(0, top_index).select(1, 0)[i].item().toFloat());
                top_ymin.push_back(batch_detections.index_select(0, top_index).select(1, 1)[i].item().toFloat());
                top_xmax.push_back(batch_detections.index_select(0, top_index).select(1, 2)[i].item().toFloat());
                top_ymax.push_back(batch_detections.index_select(0, top_index).select(1, 3)[i].item().toFloat());
            }

            //去灰条
            vector<float> boxes;
            vector<float> img_shape = {(float)frame.size().width, (float)frame.size().height};
            yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, img_shape, boxes);
            int thickness = (img_shape[0] + img_shape[1]) / IMG_SIZE;

            string predicted_class, label, score;
            int top, left, bottom, right;
            for (vector<Point2f>::size_type i = 0; i < top_label.size(); ++i)
            {
                predicted_class = class_names[top_label[i]];
                //top = max(0, (int)floor(boxes[4 * i] - 5 + 0.5));
                //left = max(0, (int)floor(boxes[4 * i + 1] - 5 + 0.5));
                //bottom = min((int)img_shape[0], (int)floor(boxes[4 * i + 2] + 5 + 0.5));
                //right = min((int)img_shape[0], (int)floor(boxes[4 * i + 3] + 5 + 0.5));
                top = max(0, (int)floor(boxes[4 * i] + 0.5));
                left = max(0, (int)floor(boxes[4 * i + 1] + 0.5));
                bottom = min((int)img_shape[1], (int)floor(boxes[4 * i + 2] + 0.5));
                right = min((int)img_shape[0], (int)floor(boxes[4 * i + 3] + 0.5));

                //画框框
                score = format("%.2f", top_conf[i]);
                label = predicted_class + " " + score;

                rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), thickness/2, 8, 0);
                int l_w = (label.length() * 8 - 5) * thickness / 2;
                int l_h = 11 * thickness / 2;
                rectangle(frame, Point(left, top - l_h), Point(left + l_w, top), Scalar(0, 0, 255), -1, 8, 0);
                putText(frame, label, Point(left, top-thickness/2), FONT_HERSHEY_SIMPLEX, 0.4*thickness/2, Scalar(0, 0, 0), thickness/2, 8);
            }
        }
        string label_fps;
        auto duration = duration_cast<microseconds>(system_clock::now() - start);
        double duration_s = (double)(duration.count()) * microseconds::period::num / microseconds::period::den;
        fps = (fps + 1. / duration_s) / 2;
        label_fps = "Fps= " + format("%.2f", fps);
        putText(frame, label_fps, Point(10, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, 8);

        imshow("yolov3", frame);
        waitKey(10);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "rosyolo");
    ros::NodeHandle n;
    ROS_INFO("Hello, YOLO!");
    RosYolov3();
    ros::spin();
}