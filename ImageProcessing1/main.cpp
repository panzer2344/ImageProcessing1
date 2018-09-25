#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int clamp(int value, int left, int right);
void generateNoise(Mat& img);
void deleteNoise(Mat& img, int radius);
void generateGaussianNoise(Mat& img, int deviation = 1, int mu = 0);

int main(int argc, char* argv[]) {
	string imageName("avto.jpg");
	Mat InputImage = imread(imageName.c_str(), IMREAD_COLOR);

	namedWindow("Input", WINDOW_AUTOSIZE);
	imshow("Input", InputImage);

	if (InputImage.empty()) {
		cout << "Could not open or find " << imageName << endl;
		return -1;
	}
	
	namedWindow("Noise", WINDOW_AUTOSIZE);
	//generateNoise(InputImage);
	generateGaussianNoise(InputImage);
	imshow("Noise", InputImage);

	namedWindow("Filtered", WINDOW_AUTOSIZE);
	deleteNoise(InputImage, 2);
	imshow("Filtered", InputImage);

	cvWaitKey(0);
	return 0;
}

void generateNoise(Mat& img) {
	srand(0);
	double probability = 0.0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			probability = (double)rand() / RAND_MAX;
			if (probability < 0.025)
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);

			if (probability > 0.975)
				img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}
}

void deleteNoise(Mat& img, int radius) {

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			
			int xLeft = (j - radius) > 0 ? (j - radius) : 0;
			int xRight = (j + radius) < (img.cols - 1) ? (j + radius) : (img.cols - 1);
			int yTop = (i - radius) > 0 ? (i - radius) : 0;
			int yBot = (i + radius) < (img.rows - 1) ? (i + radius) : (img.rows - 1);

			int max = -1, min = 255;
			Vec3b maxPixel = (0, 0, 0), minPixel = (255, 255, 255);

			for (int x = xLeft; x < xRight; x++) {
				for (int y = yTop; y < yBot; y++) {
					Vec3b pixel = img.at<Vec3b>(y, x);

					int Intensity = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];

					if (Intensity > max) {
						max = Intensity;
						maxPixel = pixel;
					}

					if (Intensity < min) {
						min = Intensity;
						minPixel = pixel;
					}

				}
			}

			img.at<Vec3b>(i, j)[0] = (maxPixel[0] + minPixel[0]) / 2;
			img.at<Vec3b>(i, j)[1] = (maxPixel[1] + minPixel[1]) / 2;
			img.at<Vec3b>(i, j)[2] = (maxPixel[2] + minPixel[2]) / 2;

		}
	}
}

void generateGaussianNoise(Mat& img, int deviation, int mu) {
	double maximum = 1 / (sqrt(2 * CV_PI) * deviation);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			double mu = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];

			for (int k = 0; k < 3; k++) {
				double result = exp(-pow((mu - pixel[k]), 2) / (2 * deviation * deviation)) / (sqrt(2 * CV_PI) * deviation);
				img.at<Vec3b>(i, j)[k] = clamp((int)(result * pixel[k] / maximum), 0, 255);
			}
		}
	}
}

int clamp(int value, int left, int right) {
	if (value > right) value = right;
	if (value < left) value = left;
	
	return value;
}