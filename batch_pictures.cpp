/**********************************************

 Copyright (c) 2020 xfmm - All rights reserved.

 NOTICE: All information contained here is, and remains
 the property of xfmm. This file can not
 be copied or distributed without permission of xfmm.

 Author: written by xfmm, 2020-1-30

 Change History:
   1. Init.

**************************************************/
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <sys/time.h>
#include "boost/filesystem.hpp"
#include<opencv2/opencv.hpp>
using namespace std;
using namespace boost::filesystem;

//将图片数据按照CHW的格式生成batch文件
int INPUT_C = 3;
int INPUT_H = 615;
int INPUT_W = 384;

//生成金字塔图时每一张图片大小和缩放因子
typedef struct dimP {
    int hs;
    int ws;
    float scale;
} dimP;

//金字塔图中每一张缩放图在金字塔图中的位置
typedef struct PositionStruct {
    float x0;
    float y0;
    float x1;
    float y1;
} Position;

void calPyramidList(cv::Mat originImage, int height, int width,  std::vector<cv::Mat> &resizeImages, std::vector<dimP> &dimention) {


    float factor = 0.506f;
    int minSize = 20;
    double scale = 12.0 / minSize; // 最小能缩放到12

    int minSide = std::min(height, width);
    minSide = minSide * scale;

    //缩放后的长度大于12
    int level = 0;
    while ((minSide >= 12) && (level < 10)) {
        int hs = std::ceil(height * scale); // 向上(大)取整数 缩放后的高
        int ws = std::ceil(width * scale); // 缩放后的宽

        dimP tmp;
        tmp.hs = hs; // 记录本次缩放后的尺度
        tmp.ws = ws;
        tmp.scale = scale; // 以及缩放因子---针对原图的缩放

        cv::Mat newImage;
        cv::resize(originImage, newImage, cv::Size(ws, hs), 0, 0, cv::INTER_NEAREST);
        //char buffer[20];
        //sprintf(buffer, "./pyramidpic%d.bmp", level);
        //cv::imwrite(buffer, newImage);

        dimention.push_back(tmp); // 尺寸
        resizeImages.push_back(newImage); // 图片

        scale = scale * factor; // 下一次的缩放比例
        minSide *= factor; // 最小边下一次缩放后的长度

        level = level + 1;
    }
}

//将缩放的图片拼成一张大图
void mergePyramidList(std::vector<cv::Mat> resizeImages, std::vector<dimP> dimention, cv::Mat &mergeImage, std::vector<Position> &position) {

    // 拼接为一个大图后的坐标
    Position tmpPosition;
    tmpPosition.x0 = 0;
    tmpPosition.y0 = 0;
    tmpPosition.x1 = 0;
    tmpPosition.y1 = -1;
    for (unsigned int i = 0; i < dimention.size(); i ++) {
        tmpPosition.x0 = 0;
        tmpPosition.x1 = dimention[i].ws;
        tmpPosition.y0 = tmpPosition.y1 + 1 ;
        tmpPosition.y1 = tmpPosition.y0 + dimention[i].hs;
        position.push_back(tmpPosition);
    }

    // 拼接为一个大图
    cv::Mat tmpImage(ceil(position.back().y1), ceil(position[0].x1), CV_32FC3, cv::Scalar::all(0));

    cv::Mat outImage;
    for (unsigned int i = 0; i < resizeImages.size(); i ++) {
        outImage = tmpImage(cv::Rect(cv::Point(position[i].x0, position[i].y0), cv::Point(position[i].x1, position[i].y1)));
        resizeImages[i].copyTo(outImage);
    }

    mergeImage = tmpImage.clone();
}

/*将数据集所有的文件写入一个txt文件中*/
int fileList(FILE* listfile,string& path) {

    boost::filesystem::path p(path.c_str());
    if(!exists(p)) {
        printf("the path %s is not existing\n", path.c_str());
        return -1;
    }
    directory_iterator end_iter;
    for(directory_iterator path_iter(p); path_iter!=end_iter; path_iter++) {
        if(is_regular_file(path_iter->status())) {
            string path = path_iter->path().string();
            fprintf(listfile, "%s\n", path.c_str());
        } else if(is_directory(path_iter->status())) {
            string path = path_iter->path().string();
            fileList(listfile, path);
        }
    }
    return 1;
}

int writeToBatchFile(string& picture, FILE* file) {

    cv::Mat image = cv::imread(picture.c_str());

    //将图片等比缩放为480*360
    double factor_w = 480/double(image.cols);
    double factor_h = 360/double(image.rows);
    double factor = min(factor_w, factor_h);
    int resize_w = std::ceil(image.cols * factor);
    resize_w = min(640, resize_w);
    int resize_h = std::ceil(image.rows * factor);
    resize_h = min(360, resize_h);

    cv::Mat resizeImage;
    cv::resize(image, resizeImage, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_NEAREST);

    cv::Mat temp(360, 640, CV_8UC3, cv::Scalar::all(0));
    resizeImage.copyTo(temp(cv::Rect(cv::Point(0,0),cv::Point(resize_w,resize_h))));
    cv::Mat data;
    temp.convertTo(data, CV_32FC3);

    //图片归一化
    auto imageSize = static_cast<std::size_t>(INPUT_H * INPUT_W * INPUT_C);
    float alpha = 0.0078125;
    float mean = 127.5;
    cv::subtract(data, cv::Scalar::all(mean), data); //cv::Scalar(alpha,alpha,alpha)
    cv::multiply(data, cv::Scalar::all(alpha), data);

    std::vector<dimP> dimention;
    // 生成金字塔图片和尺寸
    std::vector<cv::Mat> resizeImages;
    calPyramidList(data, 360, 480, resizeImages, dimention);
    // 金字塔图片合并成一张大图，和对应的位置
    cv::Mat mergeImage;
    std::vector<Position> position;
    mergePyramidList(resizeImages, dimention, mergeImage, position);

    //将图片数据存储方式由HWC转为CHW格式
    std::vector<float> imageData(imageSize);
    {
        auto data =  imageData.data();
        float *pImageR = data;
        float *pImageG = data + imageSize / 3;
        float *pImageB = data + imageSize / 3 * 2;

        for (int i = 0; i < mergeImage.rows; i++) {
            auto item = mergeImage.ptr<float>(i);

            for (int j = 0; j < mergeImage.cols; j++) {
                *pImageR++ = static_cast<float>(*item++);
                *pImageG++ = static_cast<float>(*item++);
                *pImageB++ = static_cast<float>(*item++);
            }
        }
    }
    fwrite((float *)imageData.data(), sizeof(float), imageSize, file);

}

int main() {

    //遍历数据集将所有的文件写入一个列表文件中
    FILE* file = fopen("./listfile.txt", "w+");
    string path = "./pictures/";       //数据集地址
    int ret = fileList(file, path);
    if(ret==-1) {
        cout<<"search files in the path fails"<<endl;
        return -1;
    }
    fclose(file);

    ifstream listFile;
    listFile.open("./listfile.txt", ios::in);
    if(!listFile.is_open())
        return 0;

    // image size required by mxnet Net
    static int sBatchId = 0;

    /*设置每一个patch包含的图片数*/
    int batchSize = 300;
    int batchNum = 50;
    bool end = 0;

    for(int i=0; i < batchNum; i++) {
        char buffer[50];
        sprintf(buffer, "./batches/batch%d", sBatchId++);

        FILE* batchFile = fopen(buffer, "wb+");
        if(batchFile==0) {
            cout<<errno<<std::endl;
            abort();
        }

        int s[4] = {batchSize, INPUT_C, INPUT_H, INPUT_W };
        cout<<"write header"<<std::endl;
        fwrite(s, sizeof(int), 4, batchFile);

        cout<<"write data to batch file"<<std::endl;
        string picture;
        for (int i = 0; i < batchSize; i++) {
            getline(listFile,picture);
            if(picture.empty()) {
                cout<<"reach end of the list "<<endl;
                end = true;
                break;
            }
            cout<<"picture: "<<picture.c_str()<<endl;
            int ret = writeToBatchFile(picture, batchFile);
        }
        if(end==true) {
            cout<<"reach the end"<<endl;
            fclose(batchFile);
            break;
        }
        fclose(batchFile);
    }
    listFile.close();
    return 0;
}
