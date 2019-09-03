// This is the code of 2D image affine transformation. The main character of this code is that
// it shows the relationship of spatial location between two images---- original image and affined
// image. The yellow rectangle denotes the frame of the original picture. There are 2*3 parpameters
// which represent [A,b] such that [w,v]^T=A*[x,y]^T+b, where A is 2*2 and b is 2*1.
// Note that the window shows the actual size of pictures, so make sure that the size of the window 
// will not exceed that of your computer monitor. 

#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include<iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include<math.h>
#include <opencv2/photo.hpp>


using namespace cv;
using namespace std;

//declaration 
void image_affine(Mat A);
void interpolation(double y, double x, Mat A, long* value, const char* method);

int main(int argc, char* argv)
{
	Mat img;

	//input 
	img = imread("test.jpg");
	if (!img.data)
	{
		printf("Could not load the image...\n");
		return -1;
	}

	//transformation
	image_affine(img);

	waitKey(0);
	return 1;
}


void image_affine(Mat A)
{

	double affine_mat[2][3];

	//input 
	cout << "Please input the affine parameters:" << endl;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			cin >> affine_mat[i][j];

	//compute the vertexs' location.
	Point2d vertex[4];
	vertex[0].x = 0, vertex[0].y = 0;
	vertex[1].x = A.cols - 1, vertex[1].y = 0;
	vertex[2].x = 0, vertex[2].y = A.rows - 1;
	vertex[3].x = A.cols - 1, vertex[3].y = A.rows - 1;

	for (int i = 0; i < 4; i++)
	{
		double temp = vertex[i].x * affine_mat[0][0] + vertex[i].y * affine_mat[0][1] + affine_mat[0][2];
		vertex[i].y = vertex[i].x * affine_mat[1][0] + vertex[i].y * affine_mat[1][1] + affine_mat[1][2];
		vertex[i].x = temp;
	}

	// decide the size of affined image.
	double min_x = vertex[0].x, min_y = vertex[0].y, max_x = vertex[0].x, max_y = vertex[0].y;
	for (int i = 1; i < 4; i++)
	{
		if (vertex[i].x < min_x)
			min_x = vertex[i].x;
		if (vertex[i].x > max_x)
			max_x = vertex[i].x;
		if (vertex[i].y < min_y)
			min_y = vertex[i].y;
		if (vertex[i].y > max_y)
			max_y = vertex[i].y;
	}
	double min_x_real = min_x;
	double min_y_real = min_y;
	if (min_x > 0)
		min_x = 0;
	if (min_y > 0)
		min_y = 0;

	int width, height;
	if (affine_mat[0][0] * affine_mat[1][0] > 0)
	{
		height = round(max_y - min_y);
		width = round(max_x - min_x);
	}
	else
	{
		height = round(max_y - min_y) + 1;
		width = round(max_x - min_x) + 1;
	}

	if (height < A.rows - min_y)
		height = A.rows - min_y;
	if (width < A.cols - min_x)
		width = A.cols - min_x;

	Mat B = Mat::zeros(height, width, CV_8UC3);


	//compute the inverse matrix
	double det = affine_mat[0][0] * affine_mat[1][1] - affine_mat[0][1] * affine_mat[1][0];
	double matrix_inv[2][2];
	matrix_inv[0][0] = affine_mat[1][1] / det;
	matrix_inv[0][1] = -affine_mat[0][1] / det;
	matrix_inv[1][0] = -affine_mat[1][0] / det;
	matrix_inv[1][1] = affine_mat[0][0] / det;

	//compute the inverse transformation
	double x, y, w, v;

	for (int i = 0; i < B.rows; i++)
		for (int j = 0; j < B.cols; j++)
		{
			w = j + min_x;
			v = i + min_y;

			x = matrix_inv[0][0] * (w - affine_mat[0][2]) + matrix_inv[0][1] * (v - affine_mat[1][2]);
			y = matrix_inv[1][0] * (w - affine_mat[0][2]) + matrix_inv[1][1] * (v - affine_mat[1][2]);

			long value[3] = { 0,0,0 };

			if (y >= 0 && y <= A.rows - 1 && x >= 0 && x <= A.cols - 1)
			{
				interpolation(y, x, A, value, "Bilinear"); // There is 2 methods available----"Nearest" and "Bilinear".
				B.at<Vec3b>(i, j)[0] = value[0];
				B.at<Vec3b>(i, j)[1] = value[1];
				B.at<Vec3b>(i, j)[2] = value[2];
			}
		}

	//trace the original point
	w = vertex[0].x;
	v = vertex[0].y;

	x = matrix_inv[0][0] * (w - affine_mat[0][2]) + matrix_inv[0][1] * (v - affine_mat[1][2]);
	y = matrix_inv[1][0] * (w - affine_mat[0][2]) + matrix_inv[1][1] * (v - affine_mat[1][2]);

	x = x - min_x;
	y = y - min_y;

	//display the frame of the original picture.
	Rect r(x, y, A.cols, A.rows);
	rectangle(B, r, Scalar(0, 255, 255), 2);

	//display with image fusion 
	//Mat result=B;
	//Mat imageROI =  result(Rect(int(x), int(y), A.cols, A.rows));
	//Mat mask;
	//Mat temp = 255* Mat::ones(A.rows,A.cols,A.depth());

	//addWeighted(imageROI, 0.1, temp, 0, 0., imageROI);
	//cvtColor(A, mask, COLOR_RGB2GRAY);

	//A.copyTo(imageROI, mask);	

	imshow("original image frame+ affined image", B);

}


void interpolation(double y, double x, Mat A, long* value, const char* method)
{

	int flag = 0;

	if (method == "Nearest")
	{
		flag = 1;
		y = round(y);
		x = round(x);
		if (y == A.rows)
			y = y - 1;
		if (x == A.cols)
			x = x - 1;
		value[0] = A.at<Vec3b>(y, x)[0];
		value[1] = A.at<Vec3b>(y, x)[1];
		value[2] = A.at<Vec3b>(y, x)[2];
	}

	if (method == "Bilinear")
	{
		flag = 1;
		double value_y[2][3];
		int i = 0;
		for (int j = 0; j < 3; j++)
		{
			long int a = A.at<Vec3b>(floor(y), floor(x))[j];
			long int b = A.at<Vec3b>(ceil(y), floor(x))[j];
			long int k = b - a;
			value_y[i][j] = (y - floor(y)) *  k + a;
		}
		i = 1;
		for (int j = 0; j < 3; j++)
		{
			long int a = A.at<Vec3b>(floor(y), ceil(x))[j];
			long int b = A.at<Vec3b>(ceil(y), ceil(x))[j];
			long int k = b - a;
			value_y[i][j] = (y - floor(y))  *  k + a;
		}

		for (int j = 0; j < 3; j++)
		{
			double a = value_y[0][j];
			double b = value_y[1][j];
			double k = b - a;
			value[j] = (x - floor(x)) * k + a;
		}

	}

	if (flag == 0)
	{
		cout << "We haven't got this method" << endl;
		return;
	}

}