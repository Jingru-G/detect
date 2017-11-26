/********************************************************************************************************
HOG������������ʵ��
�㷨˼·:
1)��ͼƬ�������ڴ棬��������cvtColor��ͼ��ת��ΪgrayImg
2)����һ��΢������Sobel�������ֱ�����grayImgͼ��X�����Y�����ϵ�һ��΢��/�ݶ�ͼ��
3)���ݵõ��������ݶ�ͼ��(X�����ϵ��ݶ�ͼ���Y�����ϵ��ݶ�ͼ��)��Ȼ������cartToPolar�������������
�����ݶ�ͼ������Ӧ�ĽǶȾ���ͼ��angleMat���ݶȷ�ֵ����ͼ��magnMat
4)���ǶȾ���ͼ��angleMat���������ǿ��ֵ��һ��Ϊǿ�ȷ�Χ��[0��9����9����Χ��ÿһ����Χ�ʹ���HOG��
��һ��bins
5)�ԽǶ�ΪΪ���������ݶȷ�ֵͼ�����magnMat���վŸ�������ݶȽǶȲ��Ϊ9���ݶȷ�ֵͼ�����
6)������9���Ƕȣ�ÿ���Ƕ�����Ӧ���ݶȷ�ֵͼ����󣬲�������OpenCv�еĻ��ֺ���integral�ֱ�������9
��ͼ������Ӧ�Ļ���ͼ��
==============���ˣ�����9���ݶȷ����ϣ��ֱ��Ӧ�ĵ�9���ݶȷ�ֵ����ͼ�Ѿ��������==================
7)��������ͼ����ݶȷ���ֱ��ͼHOG:Ҫ��������ͼ��ģ���Ҫ�ȼ���ÿ��Block��HOG��Ҫ����ÿ��Block��HOG
Ҫ�ȼ���ÿ��Cell��HOG
8)���㵥��Cell��HOG:����9���ݶȷ����ϵ�9���ݶȷ�ֵ����ͼ���Ѿ����������������һ���ļ���ܼ򵥣�ֻ��
Ҫ���ԼӼ����㣬����ĺ���ΪcacHOGinCell
9)���㵥��Block��HOG:�����������4��Cell��HOG���һ��Block��HOG
10)��������ͼ���HOG:��������������е�Block��HOG�ݶȷ���ֱ��ͼ������������β������һ��ά�Ⱥܴ��
����ͼ����ݶȷ���ֱ��ͼ��HOG�����������������������������ͼ����ݶȷ���ֱ��ͼ�������������
����Ҳ���Ա�����SVM�ķ���
�㷨�ѵ�:
1)����ͼ��ĸ���:�����йػ���ͼ���Blogһ���ƣ����Ǻིܶ�Ķ���׼ȷ����õİ취�ǿ�OpenCv�Ĺٷ��ĵ�
�غ����ֺ����Ľ��⣬���Խ�����ϵ����Ͽ�
2)�ѿ����ռ�����ͼ������ת��(�ؼ������һЩ����֮���໥ת����ǰ������)
3)L1������L2����:��ʹ�ù�һ��normalize����ʱ������һЩCV_L2������������L2�������Ǿ����L2�������Լ�
�����Ƶ�һ�¹�ʽ
4)����HOG�����ģ�û��ʹ�õ�����ͼ�ĸ����ʵ��HOG��ʹ�û���ͼ�������HOG�ļ����ٶȣ����ʹ���ȼ���
�ݶȣ��ڼ������������ݶȷ�����ݶȷ�ֵ�Ļ�������������̫�󣬻ᵼ��HOG�����������½�
5)���У������ÿ��Cell�Ĵ�С��20p*20p,ÿ��Block�Ĵ�СΪ4��Cell����Ȼ����������˼��Ļ���Ҳ����ʹ��
������3*3����5*5���
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
��������:
�������ͼ��
����˵��:
Mat& srcMat-----------------------�洢ÿ��cellHOG����������������
2)cv::Rect roi--------------------����cell�ľ���λ��
3)std::vector<Mat>& integrals-----�洢��9������ͼ��ÿһ������ͼ�����һ���Ƕȷ�Χ����һ��bins
*********************************************************************************************************/
// �������ͼ  
std::vector<Mat> CalculateIntegralHOG(Mat& srcMat)
{
	//��1������һ��΢�ֵ��ݶ�ͼ��  
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
	//��2������ת��,����ÿһ����X�����Y�����ϵ��ݶȣ�ʵ�ֵѿ�������ͼ������ת��  
	cartToPolar(sobelMatX, sobelMatY, magnMat, angleMat, true);
	//��3�������������д�����ʼ������ȫ����ģ���Ϊ�ڽ��ѿ�������ת��Ϊ������֮�󣬽Ƕȵķ�Χ��[0,360]  
	//     ���������д��������еĽǶ�������[0,180]�������  
	add(angleMat, Scalar(180), angleMat, angleMat<0);                //���angleMat<0�����180  
	add(angleMat, Scalar(-180), angleMat, angleMat >= 180);          //���angleMat>=180�����180  
																	 //��4���������д��뽫�ǶȾ���ת��Ϊһ���Ҷ�ֵ��Χ��[0,9]֮���ͼ��  
	angleMat /= THETA;
	//��5���������ѭ������ʵ�ǽ�ͼ����ݶȷ�ֵ���󰴾Ÿ���ͬ������ݶȽǶȣ���ÿ���Ƕȷ�Χ����Ӧ����ݶȷ�ֵ  
	//     �洢����Ӧ�ľ���ͼ��֮�ϣ���ʵ���ǽ��ݶȷ�ֵ����ͼ���ղ�ͬ���ݶȷ�ֵ�Ƕȷ�Ϊ9���ݶȷ�ֵ��ͼ��  
	for (int y = 0; y < srcMat.rows; y++)
	{
		for (int x = 0; x < srcMat.cols; x++)
		{
			int ind = angleMat.at<float>(y, x);
			bins[ind].at<float>(y, x) += magnMat.at<float>(y, x);
		}
	}
	//��6�������������ɵ�9�Ų�ͬ�Ƕȵ��ݶȷ�ֵ��������9�Ų�ͬ���ݶȷ�ֵ�Ļ���ͼ�������Ժ�  
	//     ����ͼ���ÿһ��ʹ�����һ�����Ͻǣ������ݶȷ�ֵ֮�ͣ����ɵ�9������ͼҲ����9��  
	//     bins����ͬbins�ϵ�HOGǿ��  
	std::vector<Mat> integrals(NBINS);
	for (int i = 0; i < NBINS; i++)
	{
		integral(bins[i], integrals[i]);
	}
	return integrals;
}
/********************************************************************************************************
��������:
���㵥��cell HOG����
����˵��:
1)cv::Mat& HOGCellMat-------------�洢ÿ��cellHOG����������������
2)cv::Rect roi--------------------����cell�ľ���λ��
3)std::vector<Mat>& integrals-----�洢��9������ͼ��ÿһ������ͼ�����һ���Ƕȷ�Χ����һ��bins
*********************************************************************************************************/
void cacHOGinCell(cv::Mat& HOGCellMat, cv::Rect roi, std::vector<Mat>& integrals)
{
	//��1��ͨ��9������ͼ�����ʵ��HOG�ļ��㣬HOG���ֱ��ͼ��9��bins��ÿ��bins�Ͷ�Ӧһ�Ż���ͼ��  
	int x0 = roi.x;                              //ȷ����������cell�����Ͻǵ�����  
	int y0 = roi.y;
	int x1 = x0 + roi.width;
	int y1 = y0 + roi.height;                    //ȷ����������cell�����½ǵ�����  

	for (int i = 0; i <NBINS; i++)
	{
		//��2�����ݾ��ε����Ͻǵ�����½ǵ������  
		cv::Mat integral = integrals[i];

		float a = integral.at<double>(y0, x0);
		float b = integral.at<double>(y1, x1);
		float c = integral.at<double>(y0, x1);
		float d = integral.at<double>(y1, x0);

		HOGCellMat.at<float>(0, i) = b - c - d + a;//ÿѭ��һ�Σ�����һ���ݶȷ����ϵ�HOG��������ʵ����  
												   //ÿѭ��һ�Σ��ͼ����ݶȷ���ֱ��ͼ�ϵ�һ��bins  
	}
}
/********************************************************************************************************
��������:
��ȡ��ǰ���ڵ�HOGֱ��ͼ----�˿���ʵ�����ڼ��㵥��Block��HOG�ݶȷ���ֱ��ͼ
����˵��:
1)cv::Point pt--------------------����Block�����ĵ�����
2)std::vector<cv::Mat>& integrals-----�洢��9������ͼ��ÿһ������ͼ�����һ���Ƕȷ�Χ����һ��bins
*********************************************************************************************************/
cv::Mat getHog(cv::Point pt, std::vector<cv::Mat>& integrals)
{
	if (pt.x - R<0 || pt.y - R<0 || pt.x + R >= integrals[0].cols || pt.y + R >= integrals[0].rows)
	{
		return cv::Mat();
	}
	//��1��BLOCK��HOGֱ��ͼ---�������˵��BLOCKSIZE*BLOCKSIZE��4��cell��HOG����ֱ��ͼ��������  
	//     ���һ��BLOCK��HOG����ֱ��ͼ����������  
	cv::Mat    hist(cv::Size(NBINS*BLOCKSIZE*BLOCKSIZE, 1), CV_32F);
	cv::Point  t1(0, pt.y - R);
	int c = 0;
	//��2��������:ͨ������������ѭ�����ͱ�����4��cell�����ҽ�4��cell��HOG�������������һ��  
	//     ά���Ƚϴ��BLOCK��HOG��������  
	for (int i = 0; i<BLOCKSIZE; i++)
	{
		t1.x = pt.x - R;
		for (int j = 0; j<BLOCKSIZE; j++)
		{
			//��3����ȡ��ǰ���ڣ����оֲ�HOGֱ��ͼ����  
			cv::Rect roi(t1, t1 + cv::Point(CELLSIZE, CELLSIZE));
			cv::Mat  hist_temp = hist.colRange(c, c + NBINS);
			//��4������roiȷ���ľ������򣬼��㵥��cell��HOGֱ��ͼ(�䱾�ʾ���һ������������)  
			cacHOGinCell(hist_temp, roi, integrals);
			t1.x += CELLSIZE;
			c += NBINS;
		}
		t1.y += CELLSIZE;
	}//for i  
	 //��3�����÷���2���й�һ��  
	cv::normalize(hist, hist, 1, 0, NORM_L2);
	return hist;
}
/********************************************************************************************************
��������:
��������ͼ���HOG�ݶȷ���ֱ��ͼ---HOG����
����˵��:
cv::Mat srcImage------ԭʼ�������ɫͼ��
*********************************************************************************************************/
std::vector<Mat> cacHOGFeature(cv::Mat srcImage)
{
	cv::Mat          grayImage;
	std::vector<Mat> HOGMatVector;
	cv::cvtColor(srcImage, grayImage, CV_RGB2GRAY);
	grayImage.convertTo(grayImage, CV_8UC1);
	//��1��9����ͬ�ݶȷ����ϵ�9���ݶȷ�ֵ�Ļ���ͼ�������  
	std::vector<Mat> integrals = CalculateIntegralHOG(grayImage);
	Mat image = grayImage.clone();
	image *= 0.5;
	//��2������ȫͼ�񣬼������յ��ݶȷ���ֱ��ͼHOG  
	int count = 0;
	for (int y = 0; y < grayImage.rows; y += CELLSIZE)
	{
		for (int x = 0; x < grayImage.cols; x += CELLSIZE)
		{	
			//cout << "x=" << x << "  y=" << y << endl;
			cv::Mat HOGBlockMat(Size(NBINS, 1), CV_32F);
			cv::Mat LHOGBlockMat(cv::Size(NBINS*BLOCKSIZE*BLOCKSIZE, 1), CV_32F);
			//��3����ȡ��ǰ����HOG����ʵ��ǰ�Ĵ��ھ���һ��Block��ÿ��Block���ĸ�cell��ɣ�ÿ��CellΪ20*20  
			//     �˿飬����ľ��ǵ���Block���ݶȷ���ֱ��ͼHOG  
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
			//��4��L2������һ��:����õ���ÿ��Block�ĵľ������L2������һ����ʹ��ת��Ϊһ��Block��HOG��������  
			normalize(HOGBlockMat, HOGBlockMat, 1, 0, CV_L2);
			//��5�����ÿ�õ�һ��Block��HOG���������ʹ���HOGMatVector�����HOGMatVector��ʵ��������ͼ���HOG����������  
			//     ��Ȼ���������HOGMatVector���Ǹ���ά�������ʽ�������Ҫ����SVM������з���Ļ�������Ҫ��������Ϊһ  
			//     ά��������  
			HOGMatVector.push_back(LHOGBlockMat);
			count++;
			//cout << count << endl;
			Point center(x, y);
			//��6������HOG����ͼ  
			//��ÿһ��block�����Ļ��� �߶ε���ɫ��ȱ�ʾֱ��ͼ����
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
	//��1��9����ͬ�ݶȷ����ϵ�9���ݶȷ�ֵ�Ļ���ͼ�������  
	std::vector<Mat> integrals = CalculateIntegralHOG(grayImage);
	Mat image = grayImage.clone();
	image *= 0.5;
	//��2������ȫͼ�񣬼������յ��ݶȷ���ֱ��ͼHOG  
	int count = 0;
	for (int y = 0; y < grayImage.rows; y += CELLSIZE)
	{
		for (int x = 0; x < grayImage.cols; x += CELLSIZE)
		{
			//cout << "x=" << x << "  y=" << y << endl;
			cv::Mat HOGBlockMat(Size(NBINS, 1), CV_32F);
			//��3����ȡ��ǰ����HOG����ʵ��ǰ�Ĵ��ھ���һ��Block��ÿ��Block���ĸ�cell��ɣ�ÿ��CellΪ20*20  
			//     �˿飬����ľ��ǵ���Block���ݶȷ���ֱ��ͼHOG  
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
			//��4��L2������һ��:����õ���ÿ��Block�ĵľ������L2������һ����ʹ��ת��Ϊһ��Block��HOG��������  
			normalize(HOGBlockMat, HOGBlockMat, 1, 0, CV_L2);
			//��5�����ÿ�õ�һ��Block��HOG���������ʹ���HOGMatVector�����HOGMatVector��ʵ��������ͼ���HOG����������  
			//     ��Ȼ���������HOGMatVector���Ǹ���ά�������ʽ�������Ҫ����SVM������з���Ļ�������Ҫ��������Ϊһ  
			//     ά��������  
			ImageVector.push_back(HOGBlockMat);//ImageVector�洢blockMat����55x9
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

			//��3����ȡ��ǰ����HOG����ʵ��ǰ�Ĵ��ھ���һ��Block��ÿ��Block���ĸ�cell��ɣ�ÿ��CellΪ20*20  
			//     �˿飬����ľ��ǵ���Block���ݶȷ���ֱ��ͼHOG  

			Point center(x, y);
			//��6������HOG����ͼ  
			//��ÿһ��block�����Ļ��� �߶ε���ɫ��ȱ�ʾֱ��ͼ����
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


#define MAX_PATH 1024  //�·������

static char* ImagePath = "C:\\NDS\\traindata\\man_front\\man_front_L\\";
/*----------------------------
* ���� : �ݹ�����ļ��У��ҵ����а����������ļ�
*----------------------------
* ���� : find
* ���� : public
*
* ���� : lpPath [in]      ��������ļ���Ŀ¼
* ���� : fileList [in]    ���ļ����Ƶ���ʽ�洢��������ļ�
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
	std::vector<std::string> fileList;//����һ����Ž���ļ����Ƶ�����  

											//����һ�ν���������ļ�,��ȡ�ļ����б�  
	find(ImPath, fileList);//֮��ɶ��ļ��б��е��ļ�������Ӧ�Ĳ���  
							
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
	//����ļ����������ļ�������  
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
	ofstream outFile(outfileName, ios_base::out);	//���½��򸲸Ƿ�ʽд��
	if (!outFile.is_open()) return -1;
	writeMatVector(outFile, ToatlMatVector, fileNumber);
	Draw(ToatlMatVector);
	system("pause");
	return 0;
}

 void sharpenImage1(const cv::Mat &image, cv::Mat &result)
 {
    //��������ʼ���˲�ģ��
    cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0)); 
	kernel.at<float>(1, 1) = 5.0;   
	kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;
    kernel.at<float>(2, 1) = -1.0;
    result.create(image.size(), image.type());

	   //��ͼ������˲�
    cv::filter2D(image, result, image.depth(), kernel);
 }
int AverageImage(char* ImPath)
{
	std::vector<std::string> fileList;//����һ����Ž���ļ����Ƶ�����  
	find(ImPath, fileList);//֮��ɶ��ļ��б��е��ļ�������Ӧ�Ĳ���  
	//����ļ����������ļ�������  
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
	//ofstream outFile(outfileName, ios_base::out);	//���½��򸲸Ƿ�ʽд��
	//if (!outFile.is_open()) return -1;
	//writeMatVector(outFile, ToatlMatVector);
	//Draw(ToatlMatVector, fileNumber);
	imwrite("C:\\detectProject\\data\\sourceData\\MODEL\\HOGavg_side_L.jpg", dis);
	system("pause");
	return 0;
}

/********************************************************************************************************
ģ�鹦��:
����̨Ӧ�ó�������:Main����
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
	ofstream outFile(outfileName, ios_base::out);	//���½��򸲸Ƿ�ʽд��
	if (!outFile.is_open()) return -1;
	//writeMatVector(outFile,HOGFeatureMat);

	//AverageImageHOG(ImagePath);
	cv::waitKey(0);
	return 0;
}