/********************************************************************************************************
HOG特征描述符的实现
算法思路:
1)将图片加载入内存，并且利用cvtColor将图像转换为grayImg
2)利用一阶微分算子Sobel函数，分别计算出grayImg图像X方向和Y方向上的一阶微分/梯度图像
3)根据得到的两幅梯度图像(X方向上的梯度图像和Y方向上的梯度图像)，然后利用cartToPolar函数，计算出这
两幅梯度图像所对应的角度矩阵图像angleMat和梯度幅值矩阵图像magnMat
4)将角度矩阵图像angleMat里面的像素强度值归一化为强度范围在[0，9）这9个范围，每一个范围就代表HOG中
的一个bins
5)以角度为为索引，将梯度幅值图像矩阵magnMat按照九个方向的梯度角度拆分为9幅梯度幅值图像矩阵
6)根据这9个角度，每个角度所对应的梯度幅值图像矩阵，并且利用OpenCv中的积分函数integral分别计算出这9
幅图像所对应的积分图像
==============至此，我们9个梯度方向上，分别对应的的9幅梯度幅值积分图已经计算完毕==================
7)计算整幅图像的梯度方向直方图HOG:要计算整幅图像的，需要先计算每个Block的HOG；要计算每个Block的HOG
要先计算每个Cell的HOG
8)计算单个Cell的HOG:由于9个梯度方向上的9张梯度幅值积分图像已经计算出来，所以这一步的计算很简单，只需
要记性加减计算，具体的函数为cacHOGinCell
9)计算单个Block的HOG:将计算出来的4个Cell的HOG组成一个Block的HOG
10)计算整幅图像的HOG:将计算出来的所有的Block的HOG梯度方向直方图的特征向量首尾相接组成一个维度很大的
整幅图像的梯度方向直方图的HOG特征向量，这个特征向量就是整幅图像的梯度方向直方图特征，这个特征
向量也可以被用于SVM的分类
算法难点:
1)积分图像的概念:网上有关积分图像的Blog一大推，但是很多讲的都不准确，最好的办法是看OpenCv的官方文档
关乎积分函数的讲解，可以结合网上的资料看
2)笛卡尔空间坐标和极坐标的转换(关键是理解一些它们之间相互转换的前提条件)
3)L1范数和L2范数:在使用归一化normalize函数时，考虑一些CV_L2到底是向量的L2范数还是矩阵的L2范数，自己
可以推到一下公式
4)关于HOG的论文，没有使用到积分图的概念，其实在HOG中使用积分图像加速了HOG的计算速度，如果使用先计算
梯度，在计算各个区域的梯度方向和梯度幅值的话，这样计算了太大，会导致HOG的性能有所下降
5)还有，这里的每个Cell的大小是20p*20p,每个Block的大小为4个Cell；当然如果用于行人检测的话，也可以使用
其他的3*3或者5*5组合
*********************************************************************************************************/
#include <opencv2/opencv.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/nonfree/features2d.hpp>  
#include <iostream>  
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <Windows.h>
#include <fstream>  
#include <iterator>
#include <string>

using namespace cv;
using namespace std;

#define NBINS 9  
#define THETA 180 / NBINS  
#define CELLSIZE 4  
#define BLOCKSIZE 2  
#define R (CELLSIZE * (BLOCKSIZE) * 0.5)  
#define IMAGELENGTH 400
#define IMAGEWIDTH 200
//static string ImageName = "CCHN_0018_229730_46_130718_0907_00216_Front1040.xml0";
//static string ImageName = "imageCCHN_0018_229730_46_130716_2338_00171_Front.mp4460.xml0";
string ImageName = "imageCCHN_15503_229730_46_130220_0856_00069_Front.mp4205200";
//string ImageName = "L1firstimageFILE0185190";
string ModelName = "SVM_Lw";
string ModelPath = "C:\\detectProject\\data\\sourceData\\MODEL\\\\";

/********************************************************************************************************
函数功能:
计算积分图像
参数说明:
Mat& srcMat-----------------------存储每个cellHOG特征的行特征向量
2)cv::Rect roi--------------------单个cell的矩形位置
3)std::vector<Mat>& integrals-----存储的9幅积分图像，每一幅积分图像代表一个角度范围或者一个bins
*********************************************************************************************************/
// 计算积分图  
std::vector<Mat> CalculateIntegralHOG(Mat& srcMat)
{
	//【1】计算一阶微分的梯度图像  
	cv::Mat   sobelMatX;
	cv::Mat   sobelMatY;

	cv::Sobel(srcMat, sobelMatX, CV_32F, 1, 0);
	cv::Sobel(srcMat, sobelMatY, CV_32F, 0, 1);

	std::vector<Mat> bins(NBINS);
	for (int i = 0; i < NBINS; i++)
	{
		bins[i] = Mat::zeros(srcMat.size(), CV_32F);
	}
	cv::Mat   magnMat;
	cv::Mat   angleMat;
	//【2】坐标转换,根据每一个点X方向和Y方向上的梯度，实现笛卡尔坐标和极坐标的转换  
	cartToPolar(sobelMatX, sobelMatY, magnMat, angleMat, true);
	//【3】下面这这两行代码起始是做安全处理的，因为在将笛卡尔坐标转换为极坐标之后，角度的范围在[0,360]  
	//     下面这两行代码让所有的角度收缩在[0,180]这个返回  
	add(angleMat, Scalar(180), angleMat, angleMat<0);                //如果angleMat<0，则加180  
	add(angleMat, Scalar(-180), angleMat, angleMat >= 180);          //如果angleMat>=180，则减180  
																	 //【4】下面这行代码将角度矩阵转换为一个灰度值范围在[0,9]之间的图像  
	angleMat /= THETA;
	//【5】下面这个循环，其实是将图像的梯度幅值矩阵按九个不同方向的梯度角度，将每个角度范围内相应点的梯度幅值  
	//     存储在相应的矩阵图像之上，其实就是将梯度幅值矩阵图像按照不同的梯度幅值角度分为9幅梯度幅值的图像  
	for (int y = 0; y < srcMat.rows; y++)
	{
		for (int x = 0; x < srcMat.cols; x++)
		{
			int ind = angleMat.at<float>(y, x);
			bins[ind].at<float>(y, x) += magnMat.at<float>(y, x);
		}
	}
	//【6】根据上面生成的9张不同角度的梯度幅值矩阵生成9张不同的梯度幅值的积分图像，至此以后，  
	//     积分图像的每一点就代表，这一点左上角，所有梯度幅值之和；生成的9幅积分图也就是9个  
	//     bins，不同bins上的HOG强度  
	std::vector<Mat> integrals(NBINS);
	for (int i = 0; i < NBINS; i++)
	{
		integral(bins[i], integrals[i]);
	}
	return integrals;
}
/********************************************************************************************************
函数功能:
计算单个cell HOG特征
参数说明:
1)cv::Mat& HOGCellMat-------------存储每个cellHOG特征的行特征向量
2)cv::Rect roi--------------------单个cell的矩形位置
3)std::vector<Mat>& integrals-----存储的9幅积分图像，每一幅积分图像代表一个角度范围或者一个bins
*********************************************************************************************************/
void cacHOGinCell(cv::Mat& HOGCellMat, cv::Rect roi, std::vector<Mat>& integrals)
{
	//【1】通过9幅积分图像快速实现HOG的计算，HOG这个直方图有9个bins，每个bins就对应一张积分图像  
	int x0 = roi.x;                              //确定单个矩形cell的左上角点坐标  
	int y0 = roi.y;
	int x1 = x0 + roi.width;
	int y1 = y0 + roi.height;                    //确定单个矩形cell的右下角点坐标  

	for (int i = 0; i <NBINS; i++)
	{
		//【2】根据矩形的左上角点和右下角点的坐标  
		cv::Mat integral = integrals[i];

		float a = integral.at<double>(y0, x0);
		float b = integral.at<double>(y1, x1);
		float c = integral.at<double>(y0, x1);
		float d = integral.at<double>(y1, x0);

		HOGCellMat.at<float>(0, i) = b - c - d + a;//每循环一次，计算一个梯度方向上的HOG特征，其实就是  
												   //每循环一次，就计算梯度方向直方图上的一个bins  
	}
}
/********************************************************************************************************
函数功能:
获取当前窗口的HOG直方图----此块其实就是在计算单个Block的HOG梯度方向直方图
参数说明:
1)cv::Point pt--------------------单个Block的中心点坐标
2)std::vector<cv::Mat>& integrals-----存储的9幅积分图像，每一幅积分图像代表一个角度范围或者一个bins
*********************************************************************************************************/
cv::Mat getHog(cv::Point pt, std::vector<cv::Mat>& integrals)
{
	if (pt.x - R<0 || pt.y - R<0 || pt.x + R >= integrals[0].cols || pt.y + R >= integrals[0].rows)
	{
		return cv::Mat();
	}
	//【1】BLOCK的HOG直方图---具体的来说，BLOCKSIZE*BLOCKSIZE即4个cell的HOG特征直方图特征向量  
	//     组成一个BLOCK的HOG特征直方图的特征向量  
	cv::Mat    hist(cv::Size(NBINS*BLOCKSIZE*BLOCKSIZE, 1), CV_32F);
	cv::Point  t1(0, pt.y - R);
	int c = 0;
	//【2】遍历块:通过下面这两个循环，就遍历了4个cell，并且将4个cell的HOG特征向量组成了一个  
	//     维数比较大的BLOCK的HOG特征向量  
	for (int i = 0; i<BLOCKSIZE; i++)
	{
		t1.x = pt.x - R;
		for (int j = 0; j<BLOCKSIZE; j++)
		{
			//【3】获取当前窗口，进行局部HOG直方图计算  
			cv::Rect roi(t1, t1 + cv::Point(CELLSIZE, CELLSIZE));
			cv::Mat  hist_temp = hist.colRange(c, c + NBINS);
			//【4】根据roi确定的矩形区域，计算单个cell的HOG直方图(其本质就是一个行特征向量)  
			cacHOGinCell(hist_temp, roi, integrals);
			t1.x += CELLSIZE;
			c += NBINS;
		}
		t1.y += CELLSIZE;
	}//for i  
	 //【3】利用范数2进行归一化  
	cv::normalize(hist, hist, 1, 0, NORM_L2);
	return hist;
}
/********************************************************************************************************
函数功能:
计算整幅图像的HOG梯度方向直方图---HOG特征
参数说明:
cv::Mat srcImage------原始的输入彩色图像
*********************************************************************************************************/
std::vector<Mat> cacHOGFeature(cv::Mat srcImage)
{
	cv::Mat          grayImage;
	std::vector<Mat> HOGMatVector;
	cv::cvtColor(srcImage, grayImage, CV_RGB2GRAY);
	grayImage.convertTo(grayImage, CV_8UC1);
	//【1】9个不同梯度方向上的9张梯度幅值的积分图像的生成  
	std::vector<Mat> integrals = CalculateIntegralHOG(grayImage);
	Mat image = grayImage.clone();
	image *= 0.5;
	//【2】变量全图像，计算最终的梯度方向直方图HOG  
	int count = 0;
	for (int y = 0; y < grayImage.rows; y += CELLSIZE)
	{
		for (int x = 0; x < grayImage.cols; x += CELLSIZE)
		{	
			//cout << "x=" << x << "  y=" << y << endl;
			cv::Mat HOGBlockMat(Size(NBINS, 1), CV_32F);
			cv::Mat LHOGBlockMat(cv::Size(NBINS*BLOCKSIZE*BLOCKSIZE, 1), CV_32F);
			//【3】获取当前窗口HOG，其实当前的窗口就是一个Block，每个Block由四个cell组成，每个Cell为20*20  
			//     此块，计算的就是单个Block的梯度方向直方图HOG  
			cv::Mat hist = getHog(Point(x, y), integrals);
			LHOGBlockMat = hist;
			if (hist.empty())continue;
			HOGBlockMat = Scalar(0);
			for (int i = 0; i < NBINS; i++)
			{
				for (int j = 0; j < BLOCKSIZE; j++)
				{
					HOGBlockMat.at<float>(0, i) += hist.at<float>(0, i + j*NBINS);
				}
			}
			//【4】L2范数归一化:对其得到的每个Block的的矩阵进行L2范数归一化，使其转变为一个Block的HOG特征向量  
			normalize(HOGBlockMat, HOGBlockMat, 1, 0, CV_L2);
			//【5】最后，每得到一个Block的HOG特征向量就存入HOGMatVector，这个HOGMatVector其实就是整个图像的HOG特征向量，  
			//     当然，现在这个HOGMatVector还是个二维数组的形式，如果想要利用SVM对其进行分类的话，还需要将其拉伸为一  
			//     维特征向量  
			HOGMatVector.push_back(LHOGBlockMat);
			count++;
			//cout << count << endl;
			Point center(x, y);
			//【6】绘制HOG特征图  
			//在每一个block的中心画线 线段的颜色深度表示直方图数据
			for (int i = 0; i < NBINS; i++)
			{
				double theta = (i * THETA) * CV_PI / 180.0;
				Point rd(CELLSIZE*0.5*cos(theta), CELLSIZE*0.5*sin(theta));
				Point rp = center - rd;
				Point lp = center + rd;
				line(image, rp, lp, Scalar(255 * HOGBlockMat.at<float>(0, i), 255, 255));
				cout << HOGBlockMat.at<float>(0, i)<<endl;
			}
		}
	}
	imshow("out", image);
	imwrite(ModelPath+ImageName + "HOG.jpg",image);
	return HOGMatVector;
}

std::vector<Mat> cacAvgHOGFeature(std::vector<Mat>& ToatlMatVector, cv::Mat srcImage)
{
	cv::Mat grayImage;
	std::vector<Mat> ImageVector;
	cv::cvtColor(srcImage, grayImage, CV_RGB2GRAY);
	grayImage.convertTo(grayImage, CV_8UC1);
	//【1】9个不同梯度方向上的9张梯度幅值的积分图像的生成  
	std::vector<Mat> integrals = CalculateIntegralHOG(grayImage);
	Mat image = grayImage.clone();
	image *= 0.5;
	//【2】变量全图像，计算最终的梯度方向直方图HOG  
	int count = 0;
	for (int y = 0; y < grayImage.rows; y += CELLSIZE)
	{
		for (int x = 0; x < grayImage.cols; x += CELLSIZE)
		{
			//cout << "x=" << x << "  y=" << y << endl;
			cv::Mat HOGBlockMat(Size(NBINS, 1), CV_32F);
			//【3】获取当前窗口HOG，其实当前的窗口就是一个Block，每个Block由四个cell组成，每个Cell为20*20  
			//     此块，计算的就是单个Block的梯度方向直方图HOG  
			cv::Mat hist = getHog(Point(x, y), integrals);
			if (hist.empty())continue;
			HOGBlockMat = Scalar(0);
			for (int i = 0; i < NBINS; i++)
			{
				for (int j = 0; j < BLOCKSIZE; j++)
				{
					HOGBlockMat.at<float>(0, i) += hist.at<float>(0, i + j*NBINS);
				}
			}
			//【4】L2范数归一化:对其得到的每个Block的的矩阵进行L2范数归一化，使其转变为一个Block的HOG特征向量  
			normalize(HOGBlockMat, HOGBlockMat, 1, 0, CV_L2);
			//【5】最后，每得到一个Block的HOG特征向量就存入HOGMatVector，这个HOGMatVector其实就是整个图像的HOG特征向量，  
			//     当然，现在这个HOGMatVector还是个二维数组的形式，如果想要利用SVM对其进行分类的话，还需要将其拉伸为一  
			//     维特征向量  
			ImageVector.push_back(HOGBlockMat);//ImageVector存储blockMat数据55x9
			for (int i = 0; i < NBINS; i++)
			{
				ToatlMatVector[count].at<float>(0, i) += HOGBlockMat.at<float>(0, i);
			}
			count++;
		}
	}
	//imshow("out", image);
	//imwrite("C:\\detectProject\\data\\sourceData\\MODEL\\" + ImageName + "HOG.jpg", image);
	return ToatlMatVector;
}
std::vector<Mat> Draw(std::vector<Mat> TotalMat)
{
	int count = 0;
	cv::Mat image = cv::imread(ModelPath + ModelName + ".jpg");
	for (int y = CELLSIZE; y < IMAGELENGTH-CELLSIZE; y += CELLSIZE)
	{
		for (int x = CELLSIZE; x < IMAGEWIDTH-CELLSIZE; x += CELLSIZE)
		{
			cout << "x=" << x << "  y=" << y << endl;
			cv::Mat blockMat= TotalMat[count];

			//【3】获取当前窗口HOG，其实当前的窗口就是一个Block，每个Block由四个cell组成，每个Cell为20*20  
			//     此块，计算的就是单个Block的梯度方向直方图HOG  

			Point center(x, y);
			//【6】绘制HOG特征图  
			//在每一个block的中心画线 线段的颜色深度表示直方图数据
			for (int i = 0; i < NBINS; i++)
			{
				double theta = (i * THETA) * CV_PI / 180.0;	
				Point rd(CELLSIZE*0.5*cos(theta), CELLSIZE*0.5*sin(theta));
				Point rp = center - rd;
				Point lp = center + rd;
				if (blockMat.at<float>(0, i) > 0.6)
					line(image, rp, lp, Scalar(255 * blockMat.at<float>(0, i), 255, 255));
			}
			count++;
			cout << count << endl;
		}	
	}
	//imshow("out", image);
	imwrite(ModelPath + ModelName + "HOG.jpg", image);
	return TotalMat;
}


int writeMatVector(ofstream& outfile, std::vector<Mat>& mat, int imageNumber)
{
	float data;
	int count = 0;
	for (int i = 0; i < mat.size(); i++)
	{
		for (int j = 0; j < mat[i].cols; j++)
		{
			data = mat[i].at<float>(0, j)/imageNumber;
			outfile << data << "\t";
			count++;
		}
		//cout << mat[i] << endl;

		//cout << "r(c) = " << "\n" << format(mat, "C") << " , " << endl << endl;
	}
	cout << count;
	outfile.close();
	return 0;
}


#define MAX_PATH 1024  //最长路径长度

static char* ImagePath = "C:\\NDS\\traindata\\man_front\\man_front_L\\";
/*----------------------------
* 功能 : 递归遍历文件夹，找到其中包含的所有文件
*----------------------------
* 函数 : find
* 访问 : public
*
* 参数 : lpPath [in]      需遍历的文件夹目录
* 参数 : fileList [in]    以文件名称的形式存储遍历后的文件
*/
void find(char* lpPath, std::vector<std::string> &fileList)
{
	char szFind[MAX_PATH];
	WIN32_FIND_DATA FindFileData;
	string impath = lpPath;
	impath = impath.substr(0, impath.size() - 1);
	const char* path= impath.c_str();

	strcpy(szFind, path);
	strcat(szFind, "\\*.*");

	HANDLE hFind = ::FindFirstFile(szFind, &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)    return;

	while (true)
	{
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if (FindFileData.cFileName[0] != '.')
			{
				char szFile[MAX_PATH];
				strcpy(szFile, path);
				strcat(szFile, "\\");
				strcat(szFile, (char*)(FindFileData.cFileName));
				find(szFile, fileList);
			}
		}
		else
		{
			//std::cout << FindFileData.cFileName << std::endl;
			fileList.push_back(FindFileData.cFileName);
		}
		if (!FindNextFile(hFind, &FindFileData))    break;
	}
	FindClose(hFind);
}


int AverageImageHOG(char* ImPath)
{
	std::vector<std::string> fileList;//定义一个存放结果文件名称的链表  

											//遍历一次结果的所有文件,获取文件名列表  
	find(ImPath, fileList);//之后可对文件列表中的文件进行相应的操作  
							
	std::vector<Mat> ToatlMatVector;
	for (int s = 0; s < (IMAGEWIDTH/CELLSIZE-1)*(IMAGELENGTH/CELLSIZE-1); s++)
	{
		cv::Mat tempMat(cv::Size(NBINS, 1), CV_32F);
		ToatlMatVector.push_back(tempMat);
	}
	for (int i = 0; i < (IMAGEWIDTH / CELLSIZE - 1)*(IMAGELENGTH / CELLSIZE - 1); i++)
	{
		for (int j = 0; j < NBINS; j++)
		{
			ToatlMatVector[i].at<float>(0, j) = 0.0;
		}
	}
	//输出文件夹下所有文件的名称  
	int fileNumber = 0;
	for (; fileNumber < fileList.size(); fileNumber++)
	{
		string str_pic_name = ImPath + fileList[fileNumber];
		cv::Mat srcImage = cv::imread(str_pic_name);
		if (srcImage.empty())
			return -1;
		cacAvgHOGFeature(ToatlMatVector,srcImage);
		cout << fileNumber<<"out "<< str_pic_name << endl;
	}
	string outfileName = ModelPath + ModelName + ".txt";
	ofstream outFile(outfileName, ios_base::out);	//按新建或覆盖方式写入
	if (!outFile.is_open()) return -1;
	writeMatVector(outFile, ToatlMatVector, fileNumber);
	Draw(ToatlMatVector);
	system("pause");
	return 0;
}

 void sharpenImage1(const cv::Mat &image, cv::Mat &result)
 {
    //创建并初始化滤波模板
    cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0)); 
	kernel.at<float>(1, 1) = 5.0;   
	kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;
    kernel.at<float>(2, 1) = -1.0;
    result.create(image.size(), image.type());

	   //对图像进行滤波
    cv::filter2D(image, result, image.depth(), kernel);
 }
int AverageImage(char* ImPath)
{
	std::vector<std::string> fileList;//定义一个存放结果文件名称的链表  
	find(ImPath, fileList);//之后可对文件列表中的文件进行相应的操作  
	//输出文件夹下所有文件的名称  
	int fileNumber = 0;
	cv::Mat avgImage;	
	cv::Mat dis;
	float alpha = 0.95;
	float beta = (1.0 - alpha);
	cv::Mat grayImage;
	for (; fileNumber < fileList.size(); fileNumber++)
	{

		string str_pic_name = ImPath + fileList[fileNumber];
 		cv::Mat srcImage = cv::imread(str_pic_name);
		if (srcImage.empty())
			return -1;
		cv::cvtColor(srcImage, grayImage, CV_RGB2GRAY);
		grayImage.convertTo(grayImage, CV_8UC1);
		if (fileNumber == 0)
			avgImage = grayImage;
		else
			avgImage =dis;
		cout << fileNumber << "out " << str_pic_name << endl;
		if (fileNumber == 208)
			int i = 0;
		addWeighted(avgImage, alpha, grayImage, beta, 0.0, dis);

	}
	//avgImage /= fileNumber;
	//string outfileName = "C:\\detectProject\\data\\sourceData\\MODEL\\" + ModelName + ".txt";
	//ofstream outFile(outfileName, ios_base::out);	//按新建或覆盖方式写入
	//if (!outFile.is_open()) return -1;
	//writeMatVector(outFile, ToatlMatVector);
	//Draw(ToatlMatVector, fileNumber);
	imwrite("C:\\detectProject\\data\\sourceData\\MODEL\\HOGavg_side_L.jpg", dis);
	system("pause");
	return 0;
}

/********************************************************************************************************
模块功能:
控制台应用程序的入口:Main函数
*********************************************************************************************************/

int main()
{
	cv::Mat srcImage = cv::imread(ModelPath+ImageName+".jpg");
	//imwrite("C:\\detectProject\\data\\sourceData\\MODEL\\" + ImageName + ".jpg", srcImage);
	if (srcImage.empty())
		return -1;
	cv::imshow("srcImage ", srcImage);
	std::vector<Mat> HOGFeatureMat=cacHOGFeature(srcImage);
	string outfileName = ModelPath + ImageName + ".txt";
	ofstream outFile(outfileName, ios_base::out);	//按新建或覆盖方式写入
	if (!outFile.is_open()) return -1;
	//writeMatVector(outFile,HOGFeatureMat);

	//AverageImageHOG(ImagePath);
	cv::waitKey(0);
	return 0;
}