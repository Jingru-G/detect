#include <iostream>  
#include <io.h>
#include <direct.h>
#include <fstream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <ctime>
#include <opencv2/opencv.hpp>  
#include <opencv2/nonfree/features2d.hpp>  
#include <stdio.h>
#include <vector>
#include <Windows.h>
#include <iterator>
#include <string>


using namespace std;
using namespace cv;

bool TRAIN = false;   //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��  
bool CENTRAL_CROP = false;   //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����  
							 //int TRAINTYPE = 0;


							 //�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������  
							 //��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������  
class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����  
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����  
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

class myRect
{
public:
	string group;
	double w;
	Rect rect;
};
void generateDescriptors(ifstream& imagePath, HOGDescriptor& hog, vector<float>& descriptors, int& descriptorDim,
	Mat& sampleFeatureMat, Mat& sampleLabelMat, int trainClass, int PosSamNO, int NegSamNO, int HardExampleNO) {
	string imgName;
	int numLimit;
	if (0 == trainClass)	//ѵ��������������
	{
		numLimit = PosSamNO;  //positiveSampleNumber
	}
	else if (1 == trainClass)	//ѵ�������Ǹ�����
	{
		numLimit = NegSamNO;
	}
	else if (2 == trainClass)	//ѵ������������(��)����
	{
		numLimit = HardExampleNO;
	}
	for (int num = 0; num < numLimit && getline(imagePath, imgName); num++)
	{
		//cout << imgName << endl;
		cv::Mat src = imread(imgName);//��ȡͼƬ 
		if (src.empty())
			cout<<" -1";
		namedWindow("yuanshitu", CV_WINDOW_AUTOSIZE);
		imshow("n", src);
		waitKey(30);
		cv::Mat newsrc = imread(imgName);//��ȡͼƬ  
		//CENTRAL_CROP = false;
		if (CENTRAL_CROP)
			resize(src, newsrc, hog.winSize);
		//src = src(rectCrop);//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������  
		/*		imshow("....", src);
		waitKey(6000);			*/							 //resize(src,src,Size(64,128));  
		if (cv::imwrite("C:\\detectProject\\data\\sourceData\\SAMPLE\\" + imgName + "HOG.jpg", newsrc))
			cout << "success";
		imshow("new", newsrc);
		waitKey(30);
		hog.compute(newsrc, descriptors, hog.blockStride);//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
													   //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������  
													   //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
		if (0 == trainClass)
		{
			if (0 == num)
			{
				descriptorDim = descriptors.size();	//HOG�����ӵ�ά�� 
													//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat  
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, descriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����  
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}
		else if (1 == trainClass) {
			if (0 == num)
				descriptorDim = sampleFeatureMat.cols;
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1��û��
		}
		else if (2 == trainClass)
		{
			if (0 == num)
				descriptorDim = sampleFeatureMat.cols;
			for (int i = 0; i < descriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
			sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//�����������Ϊ-1��û��
		}

	}
	descriptors.clear();
	return;
}

void trainSVM(string posPath, string negPath, string hardPath, HOGDescriptor& hog, string modelPath, vector<float>& descriptors, int PosSamNO, int NegSamNO, int HardExampleNO) {

	ifstream finPos(posPath.data());
	ifstream finNeg(negPath.data());
	ifstream finHard(hardPath.data());
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������  
	MySVM svm;//SVM������
			  //HOG����������
	string ImgName;//����·����ͼƬ��
	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��      
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����  

	cout << "��ʼ���������������" << endl;
	generateDescriptors(finPos, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 0, PosSamNO, NegSamNO, HardExampleNO);
	cout << "�������" << endl;
	cout << "��ʼ���㸺���������" << endl;
	generateDescriptors(finNeg, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 1, PosSamNO, NegSamNO, HardExampleNO);
	cout << "�������" << endl;
	if (HardExampleNO > 0)
		//���ζ�ȡHardExample���Ѹ�����ͼƬ������HOG������  
		generateDescriptors(finHard, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 2, PosSamNO, NegSamNO, HardExampleNO);

	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01  
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	cout << "��ʼѵ��SVM������" << endl;
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������  
	cout << "ѵ�����" << endl;
	svm.save(modelPath.data());//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ� 
	cout << "SVMmodel:" << modelPath << endl;
	descriptors.clear();
	finPos.close();
	finNeg.close();
	finHard.close();
	return;
}
/*******************************************************************************************************************
����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector(֧������,������)������һ�����飬����alpha,��һ��������������rho;
��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ڸ���������������һ��Ԫ��rho��
��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()��
���Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
********************************************************************************************************************/
void setDetector(MySVM& svm, vector<float>& myDetector, string detectorPath) {
	int DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��  
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���  
														  //cout << "֧������������" << supportVectorNum << endl;
	// ������Ҫ�Ķ�ά����
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//��ʼ��alphaMat��ֵȫΪ0������Ϊ1����������֧��������ά��  
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//��ʼ��֧����������  
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//��ʼ���������������洢����alpha��������֧����������Ľ��  

	//��֧�����������ݸ��Ƶ�supportVectorMat������  
	for (int i = 0; i < supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��  
		for (int j = 0; j < DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";  
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��  
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];  //alphaMatֻһ�У���һ��������
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��  
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�  
	resultMat = -1 * alphaMat * supportVectorMat;

	//��resultMat�е����ݸ��Ƶ�����myDetector��  
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������  
	myDetector.push_back(svm.get_rho());  //��vector�����µ�myDetectorһά����ĩβ���һ��rho
	cout << "�����ά����" << myDetector.size() << endl;

	//�������Ӳ������ļ�  
	ofstream fout(detectorPath.data());
	for (int i = 0; i < myDetector.size(); i++)
		fout << myDetector[i] << endl;
	fout.close();

	return;
}

double deteUpLimitS1, deteUpLimitS2, deteUpLimitM1, deteUpLimitM2, deteUpLimitL1, deteUpLimitL2;
double deteLowLimitS, deteLowLimitM, deteLowLimitL;
void DetectAndDraw(Mat& src, Mat &trtd, HOGDescriptor& hog1s, HOGDescriptor& hog1m, HOGDescriptor& hog1l, HOGDescriptor& hog2s, HOGDescriptor& hog2m, HOGDescriptor& hog2l, 
	HOGDescriptor& hog3s, HOGDescriptor& hog3m, HOGDescriptor& hog3l, vector<Rect>& found_tmp, vector<myRect>& found, vector<myRect>& found_filtered, vector<double>& weight,int plate_number_int)
{
	//string path
	//����Ƶת֡�γɵ�ͼƬ���ж�߶����˼��
	//string dirPath = "C:\\detectProject\\testdata\\";
	Rect r; 
	myRect mr;
	//cout << "here:" << plate_number << endl;	
	int deteUpLimit = 140;  int deteLowLimit = 300;
	
	switch(plate_number_int){
		// ��ͬ���ƶ�Ӧ��ͬ�ļ���������Լ�SLM�Ĳ�ͬ���ڷ�Χ
		case 229726:
		{
			deteUpLimitS1 = 176.24;	deteUpLimitS2 = 189.04;	deteLowLimitS = 227.24;
			deteUpLimitM1 = 174.36;	deteUpLimitM2 = 188.49;	deteLowLimitM = 243.36;
			deteUpLimitL1 = 166.7; deteUpLimitL2 = 194.61;	deteLowLimitL = 283.71;
		}
		case 229727:
		{
			deteUpLimitS1 = 177.24;	deteUpLimitS2 = 190.04;	deteLowLimitS = 228.24;
			deteUpLimitM1 = 175.36;	deteUpLimitM2 = 189.49;	deteLowLimitM = 243.36;
			deteUpLimitL1 = 191; deteUpLimitL2 = 218.9;	deteLowLimitL = 308;
		}
		case 229728:
		{
			deteUpLimitS1 = 150.24;	deteUpLimitS2 = 164.37;	deteLowLimitS = 219.24;
			deteUpLimitM1 = 166.36;	deteUpLimitM2 = 180.49;	deteLowLimitM = 235.36;
			deteUpLimitL1 = 166.7;	deteUpLimitL2 = 194.61;	deteLowLimitL = 283.71;
		}
		case 229729:
		{
			deteUpLimitS1 = 150.24;	deteUpLimitS2 = 164.37; deteLowLimitS = 219.24;
			deteUpLimitM1 = 166.36;	deteUpLimitM2 = 180.49;	deteLowLimitM = 235.36;
			deteUpLimitL1 = 166.7;	deteUpLimitL2 = 194.61;	deteLowLimitL = 283.71;
		}
		case 229730:
		{
			deteUpLimitS1 = 155.24; deteUpLimitS2 = 168.04; deteLowLimitS = 206.24;					
			deteUpLimitM1 = 153.36; deteUpLimitM2 = 167.49; deteLowLimitM = 222.36;				
			deteUpLimitL1 = 166.7; deteUpLimitL2 = 194.61; deteLowLimitL = 283.71;		
		}
	}

	hog1s.detectMultiScale(src(Range(deteUpLimitS1, deteLowLimitS), Range(0, 480)), found_tmp, weight, 0.15, hog1s.blockStride, Size(0, 0), 1.08, 2, false);
	//0.05~~0.1
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		// the HOG detector returns slightly larger rectangles than the real objects.  
		// so we slightly shrink the rectangles to get a nicer output.
		// cvRound() : ����������������ֵ
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 200)  // smallС����ģ��
		if (deteUpLimitS1 <= r.tl().y <= deteUpLimitS2 && r.br().y <= deteLowLimitS)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "small_front";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(),found_tmp.begin(),found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog1m.detectMultiScale(src(Range(deteUpLimitM1, deteLowLimitM), Range(0, 480)), found_tmp, weight, 0.15, hog1m.blockStride, Size(0, 0), 1.05, 2);
	// ��Բ�ͬԶ����Сģ�͵Ĳ���������ͬ
	//0.1~~0.15
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 215)  //������ģ��
		if (deteUpLimitM1 <= r.tl().y <= deteUpLimitM2 && r.br().y <= deteLowLimitM)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "middle_front";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	////found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog1l.detectMultiScale(src(Range(deteUpLimitL1, deteLowLimitL), Range(0, 480)), found_tmp, weight, 0.15, hog1l.blockStride, Size(0, 0), 1.01, 2);
	//0.2~0.25
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 230)  //������ģ��
		if (deteUpLimitL1 <= r.tl().y <= deteUpLimitL2 && r.br().y <= deteLowLimitL)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "large_front";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();

	hog2s.detectMultiScale(src(Range(deteUpLimitS1, deteLowLimitS), Range(0, 480)), found_tmp, weight, 0.15, hog2s.blockStride, Size(0, 0), 1.08, 2, false);
	//0.05~~0.1
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		// the HOG detector returns slightly larger rectangles than the real objects.  
		// so we slightly shrink the rectangles to get a nicer output.
		// cvRound() : ����������������ֵ
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 200)  // smallС����ģ��
		if (deteUpLimitS1 <= r.tl().y <= deteUpLimitS2 && r.br().y <= deteLowLimitS)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "small_side";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(),found_tmp.begin(),found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog2m.detectMultiScale(src(Range(deteUpLimitM1, deteLowLimitM), Range(0, 480)), found_tmp, weight, 0.15, hog2m.blockStride, Size(0, 0), 1.05, 2);
	// ��Բ�ͬԶ����Сģ�͵Ĳ���������ͬ
	//0.1~~0.15
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 215)  //������ģ��
		if (deteUpLimitM1 <= r.tl().y <= deteUpLimitM2 && r.br().y <= deteLowLimitM)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "middle_side";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	////found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog2l.detectMultiScale(src(Range(deteUpLimitL1, deteLowLimitL), Range(0, 480)), found_tmp, weight, 0.15, hog2l.blockStride, Size(0, 0), 1.01, 2);
	//0.2~0.25
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 230)  //������ģ��
		if (deteUpLimitL1 <= r.tl().y <= deteUpLimitL2 && r.br().y <= deteLowLimitL)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "large_side";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();

	hog3s.detectMultiScale(src(Range(deteUpLimitS1, deteLowLimitS), Range(0, 480)), found_tmp, weight, 0.15, hog3s.blockStride, Size(0, 0), 1.08, 2, false);
	//0.05~~0.1
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		// the HOG detector returns slightly larger rectangles than the real objects.  
		// so we slightly shrink the rectangles to get a nicer output.
		// cvRound() : ����������������ֵ
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 200)  // smallС����ģ��
		if (deteUpLimitS1 <= r.tl().y <= deteUpLimitS2 && r.br().y <= deteLowLimitS)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "small_ride";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(),found_tmp.begin(),found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog3m.detectMultiScale(src(Range(deteUpLimitM1, deteLowLimitM), Range(0, 480)), found_tmp, weight, 0.15, hog3m.blockStride, Size(0, 0), 1.05, 2);
	// ��Բ�ͬԶ����Сģ�͵Ĳ���������ͬ
	//0.1~~0.15
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 215)  //������ģ��
		if (deteUpLimitM1 <= r.tl().y <= deteUpLimitM2 && r.br().y <= deteLowLimitM)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "middle_ride";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	////found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();
	hog3l.detectMultiScale(src(Range(deteUpLimitL1, deteLowLimitL), Range(0, 480)), found_tmp, weight, 0.15, hog3l.blockStride, Size(0, 0), 1.01, 2);
	//0.2~0.25
	for (int i = found_tmp.size() - 1; i >= 0; i--)
	{
		r = found_tmp[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += 140;
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//if (r.tl().y <= 190 && r.br().y >= 230)  //������ģ��
		if (deteUpLimitL1 <= r.tl().y <= deteUpLimitL2 && r.br().y <= deteLowLimitL)
		{
			mr.rect = found_tmp[i];
			mr.w = weight[i];
			mr.group = "large_ride";
			found.push_back(mr);
			//found_tmp.erase(found_tmp.begin() + i);				
		}
	}
	//found.insert(found.end(), found_tmp.begin(), found_tmp.end());
	weight.clear();
	found_tmp.clear();


	//!!!!!!!!!!!!!!!!!!!!!!!!!!!�߽�ȷ��ע��(Range(300, 570), Range(0, 1280))

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��  
	int x1, x2, y1, y2;
	for (int i = 0; i < found.size(); i++)
	{
		mr = found[i];
		int j = 0;
		//for (; j < found.size(); j++)
		//	if (j != i && (r & found[j]) == r)
		//		break;
		for (; j <found.size(); j++)
		{
			x1 = cvRound((mr.rect.tl().x + mr.rect.br().x) / 2);
			x2 = cvRound((found[j].rect.tl().x + found[j].rect.br().x) / 2);
			y1 = cvRound((mr.rect.tl().y + mr.rect.br().y) / 2);
			y2 = cvRound((found[j].rect.tl().y + found[j].rect.br().y) / 2);
			if (j != i)
				if ((mr.w <= found[j].w))
					if (abs(x1 - x2) <= abs(cvRound(found[j].rect.width / 2)))
						if (abs(y1 - y2) <= abs(cvRound(found[j].rect.height / 2)))
							break;
			///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		}
		if (j == found.size())
			found_filtered.push_back(mr);
	}

	//found_filtered.insert(found_filtered.end(), found.begin(), found.end());

	string hePath;
	string headString;
	//stringstream ss;
	for (int i = 0; i < found_filtered.size(); i++)
	{
		//int x1, y1, x2, y2;
		mr = found_filtered[i];
		//x1 = cvRound(((r.x + r.br().x) - winSize.width) / 2);
		//y1 = cvRound(((r.y + r.br().y) - winSize.height) / 2);
		//y1 += 170;
		//x2 = x1 + winSize.width;
		//y2 = y1 + winSize.height;
		//if (x1 < 0) {
		//	x1 = 0;
		//	x2 = winSize.width;
		//}
		//if (x2 > 480) {
		//	x1 = 480 - winSize.width;
		//	x2 = 480;
		//}
		//if (y2 > 356){
		//	y1 = 356 - winSize.height;
		//	y2 = 356;
		//}
		/*	if (TRAINTYPE == 1)
		{
		headString = "she_";
		}
		else if(TRAINTYPE == 2)
		{
		headString = "mhe_";
		}
		else if(TRAINTYPE == 3)
		{
		headString = "bhe_";
		}*/

		//ss.str("");
		//ss << i;
		//hePath = dirPath + headString + num + "_" + ss.str() + ".jpg";
		//imwrite(hePath, src(Range(y1, y2), Range(x1, x2)));

		//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,so we slightly shrink the rectangles to get a nicer output.
		mr.rect.x += cvRound(mr.rect.width*0.1);
		mr.rect.width = cvRound(mr.rect.width*0.8);
		mr.rect.y += 140;
		mr.rect.y += cvRound(mr.rect.height*0.07);
		mr.rect.height = cvRound(mr.rect.height*0.8);
		////!!!!������ı߽��Ӧ
		//rectangle(trtd, Rect(0, 120, 480, 180), Scalar(0, 255, 0), 1);//���½�
		//rectangle(trtd, Rect(0, 190, 480, 1), Scalar(255, 255, 255), 1);//��ƽ��
		//rectangle(trtd, Rect(0, 205, 480, 1), Scalar(255, 255, 0), 1);//30m��
		//rectangle(trtd, Rect(0, 220, 480, 1), Scalar(255, 0, 255), 1);//15m��
		//rectangle(trtd, Rect(0, 235, 480, 1), Scalar(0, 0, 255), 1);//10m��


		//if(r.tl().y <190 && r.br().y>190)
		if (mr.group == "small_front")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(255, 220, 215), 1);
		if (mr.group == "middle_front")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(255, 220, 210), 1); 
		if (mr.group == "large_front")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(255, 220, 205), 1);
		if (mr.group == "small_side")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(0, 245, 255), 1);
		if (mr.group == "middle_side")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(0, 235, 250), 1);
		if (mr.group == "large_side")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(0, 225, 245), 1);
		if (mr.group == "small_ride")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(65, 200, 125), 1);
		if (mr.group == "middle_ride")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(65, 200, 135), 1);
		if (mr.group == "large_ride")
			rectangle(trtd, mr.rect.tl(), mr.rect.br(), Scalar(65, 200, 145), 1);
		if(mr.rect.tl().y>5)
			putText(trtd, mr.group.data(), cvPoint(mr.rect.tl().x, mr.rect.tl().y - 5),FONT_HERSHEY_PLAIN, 0.3, Scalar(65, 255, 255),0.05,4);

	}
	return;
}

// ͼƬתavi��Ƶ�ĺ�������������û�õ�
//void processedImgToVideo(string dirPath, char * videoPath, int tolFrame) {
//	IplImage* img;
//	string imgPath;
//	char const *fimgPath;
//	CvVideoWriter* writer = cvCreateVideoWriter(videoPath, CV_FOURCC('X', 'V', 'I', 'D'), 14, Size(480, 356));
//	stringstream ss;
//	for (int i = 0; i < tolFrame; i++)
//	{
//		ss.str("");
//		ss << i;
//		imgPath = dirPath + "pimage" + ss.str() + ".jpg";
//		fimgPath = imgPath.c_str();
//		img = cvLoadImage(fimgPath);
//		cvWriteFrame(writer, img);
//		cvReleaseImage(&img);
//		cout << imgPath << endl;
//	}
//	cvReleaseVideoWriter(&writer);
//}

//int main()
//{
//	bool bbbb = true;
//	if (bbbb == false)
//	{
//		int a = 1;
//		cout << a << endl;
//	}
//	stringstream ss;
//	int a = 100l;
//	int b = 2002;
//	ss << a;
//	cout << ss.str() << endl;
//	cout << "hhe" << endl;
//	ss.str("");
//	ss << b;
//	cout << ss.str() << endl;
//	system("pause");
//cout<< CV_VERSION<<endl;

//
//	string detectDataPath = "D:\\detectProject\\data\\sourceData\\TRAINDATA\\pvideoList.txt";
//	string sourceDataPath = "D:\\detectProject\\data\\sourceData\\TRAINDATA\\videoList.txt";
//	ifstream finDetect(detectDataPath.data());
//	ifstream finSource(sourceDataPath.data());
//	int tolFrame;
//	string detectData, sourceData, dirPath,tmpVideoPath;
//	VideoCapture cap;
//	while (getline(finDetect, detectData))
//	{
//		getline(finSource, sourceData);
//		cap.open(sourceData.data());
//		if (!cap.isOpened()) {
//			cout<<"Cannot open the video."<<sourceData<<endl;
//			return -1;
//		}
//		tolFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
//
//		dirPath =detectData + "\\";
//		tmpVideoPath =detectData + "p.avi";
//		char* videoPath = _strdup(tmpVideoPath.c_str());
//		processedImgToVideo(dirPath, videoPath,tolFrame);
//		free(videoPath);
//	}
//}
bool importData(ifstream& configFile, string& posPaths, string& negPaths, string& hardPaths, string& detectorPaths, string& modelPaths, string& posPathm,
	string& negPathm,string& hardPathm, string& detectorPathm, string& modelPathm, string& posPathl, string& negPathl,string& hardPathl, string& detectorPathl, string& modelPathl,
	int& PosSamNOs, int& NegSamNOs, int& HardExampleNOs, int& PosSamNOm, int& NegSamNOm, int& HardExampleNOm, int& PosSamNOl, int& NegSamNOl, int& HardExampleNOl)
{
	string number;
	getline(configFile, posPaths);//Сģ����������ȡ·��
	cout << "Loading posPath_S:	" << posPaths << endl;
	getline(configFile, negPaths);//Сģ�͸�������ȡ·��
	cout << "Loading negPath_S:	" << negPaths << endl;
	getline(configFile, hardPaths);//Сģ������������ȡ·��
	cout << "Loading hardPath_S:	" << hardPaths << endl;
	getline(configFile, detectorPaths);//Сģ�ͼ���Ӷ�ȡ·��
	cout << "Loading detectorPath_S:	" << detectorPaths << endl;
	getline(configFile, modelPaths);//Сģ��SVMHOG
	cout << "Loading modelPath_S:	" << modelPaths << endl;
	getline(configFile, posPathm);//��ģ����������ȡ·��
	cout << "Loading posPath_M:	" << posPathm << endl;
	getline(configFile, negPathm);//��ģ�͸�������ȡ·��
	cout << "Loading negPath_M:	" << negPathm << endl;
	getline(configFile, hardPathm);//��ģ������������ȡ·��
	cout << "Loading hardPath_M:	" << hardPathm << endl;
	getline(configFile, detectorPathm);//��ģ�ͼ���Ӷ�ȡ·��
	cout << "Loading detectorPath_M:	" << detectorPathm << endl;
	getline(configFile, modelPathm);//��ģ��SVMHOG
	cout << "Loading modelPath_M:	" << modelPathm << endl;
	getline(configFile, posPathl);//��ģ����������ȡ·��
	cout << "Loading posPath_L:	" << posPathl << endl;
	getline(configFile, negPathl);//��ģ�͸�������ȡ·��
	cout << "Loading negPath_L:	" << negPathl << endl;
	getline(configFile, hardPathl);//��ģ������������ȡ·��
	cout << "Loading hardPath_L:	" << hardPathl << endl;
	getline(configFile, detectorPathl);//��ģ�ͼ���Ӷ�ȡ·��
	cout << "Loading detectorPath_L:	" << detectorPathl << endl;
	getline(configFile, modelPathl);//��ģ��SVMHOG
	cout << "Loading modelPath_L:	" << modelPathl << endl;
	getline(configFile, number);
	PosSamNOs = stoi(number);//Сģ����������
	cout << "Loading posSamNum_S:	" << number << endl;
	getline(configFile, number);
	NegSamNOs = stoi(number);//Сģ�͸�������
	cout << "Loading negSamNum_S:	" << number << endl;
	getline(configFile, number);
	HardExampleNOs = stoi(number);//Сģ������������
	cout << "Loading hardSamNum_S:	" << number << endl;
	getline(configFile, number);
	PosSamNOm = stoi(number);//��ģ����������
	cout << "Loading posSamNum_M:	" << number << endl;
	getline(configFile, number);
	NegSamNOm = stoi(number);//��ģ�͸�������
	cout << "Loading negSamNum_M:	" << number << endl;
	getline(configFile, number);
	HardExampleNOm = stoi(number);//��ģ������������
	cout << "Loading hardSamNum_M:	" << number << endl;
	getline(configFile, number);
	PosSamNOl = stoi(number);//��ģ����������
	cout << "Loading posSamNum_L:	" << number << endl;
	getline(configFile, number);
	NegSamNOl = stoi(number);//��ģ�͸�������
	cout << "Loading negSamNum_L:	" << number << endl;
	getline(configFile, number);
	HardExampleNOl = stoi(number);//��ģ������������
	cout << "Loading hardSamNum_L:	" << number << endl;
	return true;
}


int main()
{
	string posPath1s, negPath1s, hardPath1s, detectorPath1s, modelPath1s, trainType, detectDataPath;
	string posPath1m, negPath1m, hardPath1m, detectorPath1m, modelPath1m;
	string posPath1l, negPath1l, hardPath1l, detectorPath1l, modelPath1l;//����ģ��С�д�
	string posPath2s, negPath2s, hardPath2s, detectorPath2s, modelPath2s;
	string posPath2m, negPath2m, hardPath2m, detectorPath2m, modelPath2m;
	string posPath2l, negPath2l, hardPath2l, detectorPath2l, modelPath2l;//����ģ��С�д�
	string posPath3s, negPath3s, hardPath3s, detectorPath3s, modelPath3s;
	string posPath3m, negPath3m, hardPath3m, detectorPath3m, modelPath3m;
	string posPath3l, negPath3l, hardPath3l, detectorPath3l, modelPath3l;//�ﳵģ��С�д�
	int PosSamNO1s, NegSamNO1s, HardExampleNO1s;
	int PosSamNO1m, NegSamNO1m, HardExampleNO1m;
	int	PosSamNO1l, NegSamNO1l, HardExampleNO1l;
	int PosSamNO2s, NegSamNO2s, HardExampleNO2s;
	int PosSamNO2m, NegSamNO2m, HardExampleNO2m;
	int	PosSamNO2l, NegSamNO2l, HardExampleNO2l;
	int PosSamNO3s, NegSamNO3s, HardExampleNO3s;
	int PosSamNO3m, NegSamNO3m, HardExampleNO3m;
	int	PosSamNO3l, NegSamNO3l, HardExampleNO3l;
	Size winSize1, blockSize1, blockStride1, cellSize1;
	Size winSize2, blockSize2, blockStride2, cellSize2;
	Size winSize3, blockSize3, blockStride3, cellSize3;
	Rect rectCrop1;
	Rect rectCrop2;
	Rect rectCrop3;
	//������������������������HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������  
	//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ  
	//if (1 == TRAINTYPE)
	//{
	//posPath = "D:\\detectProject\\SmallTrainData.txt";//������ͼƬ���ļ����б�
	//negPath = "D:\\detectProject\\NegativeData1.txt";//������ͼƬ���ļ����б�
	//hardPath = "";
	//modelPath = "D:\\detectProject\\model\\SVM_HOG_S.xml";
	//detectorPath = "D:\\detectProject\\model\\HOGDetector_S.txt";  

	winSize1 = Size(16, 32);
	blockSize1 = Size(4, 4);
	blockStride1 = Size(2, 2);
	cellSize1 = Size(2, 2);
	rectCrop1 = Rect(0, 0, 16, 32);
	//winSize = Size(48, 96);
	//blockSize = Size(16, 16);
	//blockStride = Size(8, 8);
	//cellSize = Size(8, 8);
	//rectCrop = Rect(1, 2, 48, 96);
	//}
	//else if (2 == TRAINTYPE)
	//{
	//posPath = "D:\\detectProject\\MiddleTrainData.txt";//������ͼƬ���ļ����б�
	//negPath = "D:\\detectProject\\NegativeData2.txt";//������ͼƬ���ļ����б�
	//hardPath = "";
	//modelPath = "D:\\detectProject\\model\\SVM_HOG_M.xml";
	//detectorPath = "D:\\detectProject\\model\\HOGDetector_M.txt";
	winSize2 = Size(24, 48);
	blockSize2 = Size(8, 8);
	blockStride2 = Size(4, 4);
	cellSize2 = Size(4, 4);
	rectCrop2 = Rect(0, 1, 24, 48);
	//winSize = Size(96, 192);
	//blockSize = Size(16, 16);
	//blockStride = Size(8, 8);
	//cellSize = Size(8, 8);
	//rectCrop = Rect(2, 4, 96, 192);

	/*}
	else if (3 == TRAINTYPE)
	{*/
	//posPath = "D:\\detectProject\\LargeTrainData.txt";//������ͼƬ���ļ����б�
	//negPath = "D:\\detectProject\\NegativeData3.txt";//������ͼƬ���ļ����б�
	//hardPath = "";
	//modelPath = "D:\\detectProject\\model\\SVM_HOG_L.xml";
	//detectorPath = "D:\\detectProject\\model\\HOGDetector_L.txt";
	winSize3 = Size(48, 96);
	blockSize3 = Size(16, 16);
	blockStride3 = Size(8, 8);
	cellSize3 = Size(8, 8);
	rectCrop3 = Rect(1, 2, 48, 96);
	//winSize = Size(192, 384);
	//blockSize = Size(16, 16);
	//blockStride = Size(8, 8);
	//cellSize = Size(8, 8);
	//rectCrop = Rect(4, 8, 192, 384);
	//}
	//TRAIN = true;
	HOGDescriptor hog1s(winSize1, blockSize1, blockStride1, cellSize1, 9);
	HOGDescriptor hog1m(winSize2, blockSize2, blockStride2, cellSize2, 9);
	HOGDescriptor hog1l(winSize3, blockSize3, blockStride3, cellSize3, 9);
	HOGDescriptor hog2s(winSize1, blockSize1, blockStride1, cellSize1, 9);
	HOGDescriptor hog2m(winSize2, blockSize2, blockStride2, cellSize2, 9);
	HOGDescriptor hog2l(winSize3, blockSize3, blockStride3, cellSize3, 9);
	HOGDescriptor hog3s(winSize1, blockSize1, blockStride1, cellSize1, 9);
	HOGDescriptor hog3m(winSize2, blockSize2, blockStride2, cellSize2, 9);
	HOGDescriptor hog3l(winSize3, blockSize3, blockStride3, cellSize3, 9);
	string configPath = ".\\ndsconfig.txt";
	ifstream configFile(configPath.data());
	importData(configFile, posPath1s, negPath1s, hardPath1s, detectorPath1s, modelPath1s, posPath1m, negPath1m, hardPath1m, detectorPath1m, modelPath1m, posPath1l, negPath1l, hardPath1l, 
		detectorPath1l, modelPath1l, PosSamNO1s, NegSamNO1s, HardExampleNO1s, PosSamNO1m, NegSamNO1m, HardExampleNO1m, PosSamNO1l, NegSamNO1l, HardExampleNO1l);//��������ģ�Ͳ���
	importData(configFile, posPath2s, negPath2s, hardPath2s, detectorPath2s, modelPath2s, posPath2m, negPath2m, hardPath2m, detectorPath2m, modelPath2m, posPath2l, negPath2l, hardPath2l,
		detectorPath2l, modelPath2l, PosSamNO2s, NegSamNO2s, HardExampleNO2s, PosSamNO2m, NegSamNO2m, HardExampleNO2m, PosSamNO2l, NegSamNO2l, HardExampleNO2l);//�������ģ�Ͳ���
	importData(configFile, posPath3s, negPath3s, hardPath3s, detectorPath3s, modelPath3s, posPath3m, negPath3m, hardPath3m, detectorPath3m, modelPath3m, posPath3l, negPath3l, hardPath3l,
		detectorPath3l, modelPath3l, PosSamNO3s, NegSamNO3s, HardExampleNO3s, PosSamNO3m, NegSamNO3m, HardExampleNO3m, PosSamNO3l, NegSamNO3l, HardExampleNO3l);//�����ﳵģ�Ͳ���
	getline(configFile, detectDataPath);//������Ƶ����λ��  ...videoList.txt
	cout << "Loading detectDataPath:	" << detectDataPath << endl;
	getline(configFile, trainType);   //ngconfig�ĵ����ڶ�������  
	if (trainType == "1")
		TRAIN = true;
	//cout << "TRAIN��ֵ��:"<<TRAIN << endl;
	cout << "Loading isTrain:	" << trainType << endl;
	getline(configFile, trainType);
	if (trainType == "1")
		CENTRAL_CROP = true;
	cout << "Loading isCrop:		" << trainType << endl;
	configFile.close();
	//getline(configFile, trainType);
	//TRAINTYPE = stoi(trainType);
	//cout << "Loading trainType:	" << trainType << endl;
		                                                                                                                                                                                                                                                                                  
	vector<float> descriptors;
	if (TRAIN == true)
	{
		trainSVM(posPath1s, negPath1s, hardPath1s, hog1s, modelPath1s, descriptors, PosSamNO1s, NegSamNO1s, HardExampleNO1s);
		trainSVM(posPath1m, negPath1m, hardPath1m, hog1m, modelPath1m, descriptors, PosSamNO1m, NegSamNO1m, HardExampleNO1m);
		trainSVM(posPath1l, negPath1l, hardPath1l, hog1l, modelPath1l, descriptors, PosSamNO1l, NegSamNO1l, HardExampleNO1l);
		trainSVM(posPath2s, negPath2s, hardPath2s, hog2s, modelPath2s, descriptors, PosSamNO2s, NegSamNO2s, HardExampleNO2s);
		trainSVM(posPath2m, negPath2m, hardPath2m, hog2m, modelPath2m, descriptors, PosSamNO2m, NegSamNO2m, HardExampleNO2m);
		trainSVM(posPath2l, negPath2l, hardPath2l, hog2l, modelPath2l, descriptors, PosSamNO2l, NegSamNO2l, HardExampleNO2l);
		trainSVM(posPath3s, negPath3s, hardPath3s, hog3s, modelPath3s, descriptors, PosSamNO3s, NegSamNO3s, HardExampleNO3s);
		trainSVM(posPath3m, negPath3m, hardPath3m, hog3m, modelPath3m, descriptors, PosSamNO3m, NegSamNO3m, HardExampleNO3m);
		trainSVM(posPath3l, negPath3l, hardPath3l, hog3l, modelPath3l, descriptors, PosSamNO3l, NegSamNO3l, HardExampleNO3l);
	}

	MySVM svm1s, svm1m, svm1l, svm2s, svm2m, svm2l, svm3s, svm3m, svm3l;
	vector<float> myDetector;

	svm1s.load(modelPath1s.data());
	setDetector(svm1s, myDetector, detectorPath1s);
	hog1s.setSVMDetector(myDetector);
	myDetector.clear();

	svm1m.load(modelPath1m.data());
	setDetector(svm1m, myDetector, detectorPath1m);
	hog1m.setSVMDetector(myDetector);
	myDetector.clear();

	svm1l.load(modelPath1l.data());
	setDetector(svm1l, myDetector, detectorPath1l);
	hog1l.setSVMDetector(myDetector);
	myDetector.clear();

	svm2s.load(modelPath2s.data());
	setDetector(svm2s, myDetector, detectorPath2s);
	hog2s.setSVMDetector(myDetector);
	myDetector.clear();

	svm2m.load(modelPath2m.data());
	setDetector(svm1m, myDetector, detectorPath2m);
	hog2m.setSVMDetector(myDetector);
	myDetector.clear();

	svm2l.load(modelPath2l.data());
	setDetector(svm2l, myDetector, detectorPath2l);
	hog2l.setSVMDetector(myDetector);
	myDetector.clear();

	svm3s.load(modelPath3s.data());
	setDetector(svm3s, myDetector, detectorPath3s);
	hog3s.setSVMDetector(myDetector);
	myDetector.clear();

	svm3m.load(modelPath3m.data());
	setDetector(svm3m, myDetector, detectorPath3m);
	hog3m.setSVMDetector(myDetector);
	myDetector.clear();

	svm3l.load(modelPath3l.data());
	setDetector(svm3l, myDetector, detectorPath3l);
	hog3l.setSVMDetector(myDetector);
	myDetector.clear();

	/**************����ͼƬ����HOG���˼��******************/
	cout << "Start Detecting..." << endl;
	vector<Rect> found_tmp;//���ο�����
	vector<myRect> found_filtered, found;
	vector<double> weight;
	ifstream finDetect(detectDataPath.data());	
	//cout << detectDataPath;   //.../videoLists.txt
	string detectData, videoPath, rectFilePath;
	Mat src, trtd;
	IplImage* iplimage;  //ͼƬת��Ƶ�õ��ı���
	string imgPath;
	stringstream ss;
	VideoCapture cap;
	CvVideoWriter* writer;
	double totalFrame;
	vector <string> splitString1;
	vector <string> splitString2;

	while (getline(finDetect, detectData))
	{
		cout << "Detecting " << detectData << endl;
		videoPath = detectData;
		cap.open(videoPath.data());   //�����Բ��Ե�videos�ļ���
		if (!cap.isOpened()) {
			cout << "Cannot open the video whose path is " << videoPath << endl;
			continue;
		}
		// string ---> char *
		char *path_video = (char *)videoPath.c_str();
		// ��'\'�ָ���ַ���
		const char *split = "\\";
		char *p = strtok(path_video, split);
		while (p != NULL) {
			splitString1.push_back(p);
			//nums[i] = p;
			p = strtok(NULL, split);
		}
		string videoname = splitString1[splitString1.size() - 1];  //��'\'�ָ�����һ�������Ƶ�ļ���ȫ��
		cout << videoname << endl;
		char *name_video = (char *)videoname.c_str();
		// ��'\'�ָ���ַ���
		const char *split2 = "_";
		char *p2 = strtok(name_video, split2);
		while (p2 != NULL) {
			splitString2.push_back(p2);
			//nums[i] = p;
			p2 = strtok(NULL, split2);
		}
		string plate_number = splitString2[2];  // ���ƺŶ�ȡ���
		int plate_number_int = atoi(plate_number.c_str());
		//cout << plate_number_int << endl;
		totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);   //��ȡ֡��
		//*******
		videoPath = detectData.substr(0, detectData.length() - 4) + "p.avi";  
		//����Ƶ���ӿ��Ľ����Ƶ���·��������,���ļ���һ�����ļ���һ�»�������.mp4�滻Ϊp.avi
		rectFilePath = detectData.substr(0, detectData.length() - 4) + "r.txt";  // ������ļ����·��������
		ofstream fout(rectFilePath.data());
		writer = cvCreateVideoWriter(videoPath.data(), CV_FOURCC('X', 'V', 'I', 'D'), 14, Size(480, 356));  //д�����Ƶ�������		
		for (int num = 0; num<totalFrame; num++) {
			ss.str("");
			ss << num;
			cap.read(src);
			trtd = src.clone();
			//cout << "there:" << videoPath << endl;
			DetectAndDraw(src, trtd, hog1s, hog1m, hog1l, hog2s, hog2m, hog2l, hog3s, hog3m, hog3l, found_tmp, found, found_filtered, weight, plate_number_int);
			//detectData.substr(0, detectData.length() - 4) +"_"+ss.str()
			/*if (_access((detectData.substr(0, detectData.length() -4)).data(), 0) == -1) {
			_mkdir((detectData.substr(0, detectData.length() - 4)).data());
			cout << detectData.substr(0, detectData.length() - 4) << endl;
			}*/

			iplimage = &IplImage(trtd); //д�����Ƶ�������
			cvWriteFrame(writer, iplimage); //д�����Ƶ����
			//			cvReleaseImage(&iplimage);

			//imgPath = detectData.substr(0, detectData.length() - 4) + "\\pimage" + ss.str() + ".jpg";
			for (int i = 0; i < found_filtered.size(); i++)
			{
				fout << found_filtered[i].rect.tl().x << " " << found_filtered[i].rect.tl().y << " "
					<< found_filtered[i].rect.br().x << " " << found_filtered[i].rect.br().y << " "
					<< found_filtered[i].group << ",";
			}
			fout << endl;
			found.clear();
			found_tmp.clear();
			weight.clear();
			found_filtered.clear();
			//imwrite(imgPath, trtd);
		}
		fout.close();
		 cvReleaseVideoWriter(&writer);  //д�����Ƶ�������
		cap.release();
	}
	finDetect.close();

	//	namedWindow("src", 0);
	//	imshow("src", trtd);
	//	waitKey();//ע�⣺imshow֮������waitKey�������޷���ʾͼ��  

	cout << "completed!" << endl;
	system("pause");
}

//���ζ�ȡ������ͼƬ������HOG������  
//for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
//{
//	//cout << "����" << ImgName << num << endl;
//	ImgName = "D:\\detectProject\\traindata\\" + ImgName;//������������·����  
//	Mat src = imread(ImgName);//��ȡͼƬ  
//	//imshow("....", src);
//	//waitKey(6000);   //��imshow֮�����û��waitKey����򲻻�������ʾͼ��
//	if (CENTRAL_CROP)
//		src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128��Ŀ��������Ͻ�����(16,16)�����64�����128
//										 //resize(src,src,Size(64,128));  
//	hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//											  //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������  
//	cout << descriptors.size() << endl;
//	if (0 == num)
//	{
//		DescriptorDim = descriptors.size();//HOG�����ӵ�ά��  
//										   //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat  
//		sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
//		//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����  
//		sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
//	}
//	//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
//	sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
//	descriptors.clear();
//}

////���ζ�ȡ������ͼƬ������HOG������  
//for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
//{
//	//cout << "����" << ImgName << num << endl;
//	ImgName = "D:\\detectProject\\negativedata\\" + ImgName;//���ϸ�������·����  
//	Mat src = imread(ImgName);//��ȡͼƬ  
//							  //resize(src,img,Size(64,128));  
//	//imshow("....", src);
//	//waitKey(6000);
//	hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//											  //cout<<"������ά����"<<descriptors.size()<<endl;  
//											  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��  
//	sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1������  
//	descriptors.clear();
//}

//for (int num = 0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
//{
//	cout << "����" << ImgName << endl;
//	ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����  
//	Mat src = imread(ImgName);//��ȡͼƬ  
//							  //resize(src,img,Size(64,128));  
//	hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//											  //cout<<"������ά����"<<descriptors.size()<<endl; 
//											  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
//	for (int i = 0; i < DescriptorDim; i++)
//		sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��  
//	sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������  
//	descriptors.clear();
//}


////��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9  
//HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9);//HOG���������������HOG�����ӵ�  
//int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������  
//MySVM svm;//SVM������
//vector<float> descriptors;//HOG����������
////namedWindow("~.~");
//		  //��TRAINΪtrue������ѵ��������  
//if (TRAIN)
//{
//	string ImgName;//ͼƬ��(����·��)  
//	ifstream finPos("D:\\detectProject\\LargeTrainData.txt");//������ͼƬ���ļ����б�  
//	ifstream finNeg("D:\\detectProject\\NegativeData3.txt");//������ͼƬ���ļ����б�  

//	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��      
//	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����  

//	string trainPath = "D:\\detectProject\\traindata\\";
//	string bgPath = "D:\\detectProject\\negativedata\\";
//	//���ζ�ȡ������ͼƬ������HOG������  
//	generateDescriptors(finPos, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 0, trainPath);
//	//���ζ�ȡ������ͼƬ������HOG������  
//	generateDescriptors(finNeg, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 1, bgPath);
//	
//	//����HardExample������  
//	if (HardExampleNO > 0)
//	{
//		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample������ͼƬ���ļ����б�
//		string hardPath = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\";
//		generateDescriptors(finHardExample, hog, descriptors, DescriptorDim, sampleFeatureMat, sampleLabelMat, 2, hardPath);																	 //���ζ�ȡHardExample������ͼƬ������HOG������  
//	}

//	////���������HOG�������������ļ�  
//	/*ofstream fout("D:\\detectProject\\SampleFeatureMat.txt");  
//	for(int i=0; i<PosSamNO+NegSamNO; i++)  
//	{  
//	  fout<<i<<endl;  
//	  for(int j=0; j<DescriptorDim; j++)  
//	      fout<<sampleFeatureMat.at<float>(i,j)<<"  ";  
//	  fout<<endl;  
//	} */ 

//	//ѵ��SVM������  
//	//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����  
//	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
//	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01  
//	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
//	cout << "��ʼѵ��SVM������" << endl;
//	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������  
//	cout << "ѵ�����" << endl;
//	svm.save("D:\\detectProject\\model\\SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  
//}
//else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����  
//{
//	svm.load("D:\\detectProject\\model\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��  
//}


//int DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��  
//int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���  
////cout << "֧������������" << supportVectorNum << endl;

//Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������  
//Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������  
//Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��  

//													   //��֧�����������ݸ��Ƶ�supportVectorMat������  
//for (int i = 0; i < supportVectorNum; i++)
//{
//	const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��  
//	for (int j = 0; j < DescriptorDim; j++)
//	{
//		//cout<<pData[j]<<" ";  
//		supportVectorMat.at<float>(i, j) = pSVData[j];
//	}
//}

////��alpha���������ݸ��Ƶ�alphaMat��  
//double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
//for (int i = 0; i < supportVectorNum; i++)
//{
//	alphaMat.at<float>(0, i) = pAlphaData[i];
//}

////����-(alphaMat * supportVectorMat),����ŵ�resultMat��  
////gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�  
//resultMat = -1 * alphaMat * supportVectorMat;

////�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����  
//vector<float> myDetector;
////��resultMat�е����ݸ��Ƶ�����myDetector��  
//for (int i = 0; i < DescriptorDim; i++)
//{
//	myDetector.push_back(resultMat.at<float>(0, i));
//}
////������ƫ����rho���õ������  
//myDetector.push_back(svm.get_rho());
//cout << "�����ά����" << myDetector.size() << endl;
////����HOGDescriptor�ļ����  
//hog.setSVMDetector(myDetector);
////myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());  

////�������Ӳ������ļ�  
//ofstream fout("D:\\detectProject\\HOGDetectorForOpenCV.txt");
//for (int i = 0; i < myDetector.size(); i++)
//{
//	fout << myDetector[i] << endl;
//}

/******************���뵥��64*128�Ĳ���ͼ������HOG�����ӽ��з���*********************/
////��ȡ����ͼƬ(64*128��С)����������HOG������  
////Mat testImg = imread("person014142.jpg");  
//Mat testImg = imread("noperson000026.jpg");  
//vector<float> descriptor;  
//hog.compute(testImg,descriptor,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//����������������������  
////������õ�HOG�����Ӹ��Ƶ�testFeatureMat������  
//for(int i=0; i<descriptor.size(); i++)  
//  testFeatureMat.at<float>(0,i) = descriptor[i];  

////��ѵ���õ�SVM�������Բ���ͼƬ�������������з���  
//int result = svm.predict(testFeatureMat);//�������  
//cout<<"��������"<<result<<endl;  

////cout << "���ж�߶�HOG������" << endl;
//hog.detectMultiScale(src(Range(300, 720), Range(0, 1280)), found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//��ͼƬ���ж�߶����˼��  
////!!!!!!!!!!!!!!!!!!!!!!!!!!!�߽�ȷ��ע��
////cout << "�ҵ��ľ��ο������" << found.size() << endl;

////�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��  
//for (int i = 0; i < found.size(); i++)
//{
//	Rect r = found[i];
//	int j = 0;
//	for (; j < found.size(); j++)
//		if (j != i && (r & found[j]) == r)
//			break;
//	if (j == found.size())
//		found_filtered.push_back(r);
//}
////�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����  
//for (int i = 0; i < found_filtered.size(); i++)
//{
//	Rect r = found_filtered[i];
//	r.x += cvRound(r.width*0.1);
//	r.width = cvRound(r.width*0.8);
//	r.y += cvRound(r.height*0.07);
//	r.y += 300;
//	//!!!!������ı߽��Ӧ
//	r.height = cvRound(r.height*0.8);
//	rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
//}