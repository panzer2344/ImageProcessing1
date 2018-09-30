#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int clamp(int value, int left, int right); // closing a number in the border between high and low

void generateNoise(Mat& img);
void deleteNoise(Mat& img, int radius);
void generateGaussianNoise(Mat& img, int deviation = 6, int mu = 0);

double luminance(double muX, double muY);
double contrast(double sigmaX, double sigmaY);
double structure(double sigmaXY, double sigmaX, double sigmaY);

double ssim(double _luminance, double _contrast, double _structure, double alfa = 1, double beta = 1, double gama = 1);

double average(Mat& img);
double variance(Mat& img, double muX);
double covariance(Mat& img1, Mat& img2, double muX, double muY);

void calculateAndPrintParametres(Mat& img1, Mat& img2); // calculation of characteristics of an image and print it on the console


int main(int argc, char* argv[]) {
	string imageName("avto.jpg");
	Mat InputImage = imread(imageName.c_str(), IMREAD_COLOR);
	Mat SaltAndPaperNoiseImage = InputImage.clone();
	Mat GaussianNoiseImage = InputImage.clone();
	Mat FilteredSPNImage, FilteredGImage; // SPN = salt and paper, G = gaussian

	// display the original image
	namedWindow("Input", WINDOW_AUTOSIZE);
	imshow("Input", InputImage);
	moveWindow("Input", 0, 0);

	if (InputImage.empty()) {
		cout << "Could not open or find " << imageName << endl;
		return -1;
	}

	// display the salt and paper noise image
	namedWindow("Salt and paper noise", WINDOW_AUTOSIZE);
	generateNoise(SaltAndPaperNoiseImage);
	imshow("Salt and paper noise", SaltAndPaperNoiseImage);
	moveWindow("Salt and paper noise", SaltAndPaperNoiseImage.cols, 0);

	// display the gaussian noise image
	namedWindow("Gaussian noise", WINDOW_AUTOSIZE);
	generateGaussianNoise(GaussianNoiseImage);
	imshow("Gaussian noise", GaussianNoiseImage);
	moveWindow("Gaussian noise", GaussianNoiseImage.cols, GaussianNoiseImage.rows);

	// display the salt and paper filtered image
	FilteredSPNImage = SaltAndPaperNoiseImage.clone();
	namedWindow("Filtered salt and paper noise", WINDOW_AUTOSIZE);
	deleteNoise(FilteredSPNImage, 2);
	imshow("Filtered salt and paper noise", FilteredSPNImage);
	moveWindow("Filtered salt and paper noise", 2 * FilteredSPNImage.cols, 0);

	// display the filtered gaussian image
	FilteredGImage = GaussianNoiseImage.clone();
	namedWindow("Filtered gaussian noise", WINDOW_AUTOSIZE);
	deleteNoise(FilteredGImage, 2);
	imshow("Filtered gaussian noise", FilteredGImage);
	moveWindow("Filtered gaussian noise", 2 * FilteredGImage.cols, FilteredGImage.rows);

	/*																		*/				
	/* calculation of characteristics of an image and print it on the console */
	/*																		*/																		


	/*					salt and paper										*/
	
	// for original and salt and paper noise images
	cout << endl << "*** original and SAP noise ***" << endl;
	calculateAndPrintParametres(InputImage, SaltAndPaperNoiseImage);

	// for salt and paper noise and filtered salt and paper images
	cout << endl << "*** SAP noise and filtered SAP ***" << endl;
	calculateAndPrintParametres(SaltAndPaperNoiseImage, FilteredSPNImage);

	// for original and filtered salt and paper images
	cout << endl << "*** original and filtered SAP noise ***" << endl;
	calculateAndPrintParametres(InputImage, FilteredSPNImage);

	/*					gaussian											*/					
	
	// for original and gaussian noise images
	cout << endl << "*** original and gaussian noise ***" << endl;
	calculateAndPrintParametres(InputImage, GaussianNoiseImage);

	// for gaussian noise and filtered gaussian images
	cout << endl << "*** gaussian noise and filtered gaussian ***" << endl;
	calculateAndPrintParametres(GaussianNoiseImage, FilteredGImage);

	//for original and filtered gaussian images
	cout << endl << "*** original and filtered gaussian ***" << endl;
	calculateAndPrintParametres(InputImage, FilteredGImage);

	cvWaitKey(0);
	destroyAllWindows();
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
			// find the edge neighborhood of a point
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
			img.at<Vec3b>(i, j)[0] = (maxPixel[0] + minPixel[0]) / 2;  // avg between max and min
			img.at<Vec3b>(i, j)[1] = (maxPixel[1] + minPixel[1]) / 2;
			img.at<Vec3b>(i, j)[2] = (maxPixel[2] + minPixel[2]) / 2;
		}
	}
}

/*void generateGaussianNoise(Mat& img, int deviation, int mu) {
	double maximum = 1 / (sqrt(2 * CV_PI) * deviation);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			double grayLevel = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
			long double result = exp(-pow((long double)(grayLevel - mu), 2) / (long double)(2 * deviation * deviation)) / (sqrt(2 * CV_PI) * deviation);

			for (int k = 0; k < 3; k++) {
				img.at<Vec3b>(i, j)[k] = clamp((int)result + pixel[k], 0, 255);
			}
		}
	}
}*/

void generateGaussianNoise(Mat& img, int deviation, int mu) {
	srand(0);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			double prob = (double)rand() / RAND_MAX;
			int addGrayLevel = (int)( deviation * ( -2 * log( prob * sqrt(2 * CV_PI) * deviation) ) ); // the gray level derived from the normal distribution formula

			for (int k = 0; k < 3; k++) {
				img.at<Vec3b>(i, j)[k] = clamp(pixel[k] + addGrayLevel, 0, 255); // gaussian noise is additional noise
			}
		}
	}
}



int clamp(int value, int left, int right) {
	int result = value;
	
	if (result > right) result = right;
	if (result < left) result = left;
	
	return result;
}

double luminance(double muX, double muY) {
	return 2 * muX * muY / (muX * muX + muY * muY); // luminance formula
}

double contrast(double sigmaX, double sigmaY) {
	return 2 * sigmaX * sigmaY / (sigmaX * sigmaX + sigmaY * sigmaY); // contrast formula
}

double structure(double sigmaXY, double sigmaX, double sigmaY) {
	return sigmaXY / (sigmaX * sigmaY); // structure formula
}

double ssim(double _luminance, double _contrast, double _structure, double alfa, double beta, double gama) {
	double poweredLuminance = _luminance, poweredContrast = _contrast, poweredStructure = _structure;
	
	// mini optimisation. try to not use pow func when it isn't needed
	if (alfa != 1) poweredLuminance = pow(_luminance, alfa);
	if (beta != 1) poweredContrast = pow(_contrast, beta);
	if (gama != 1) poweredStructure = pow(_structure, gama);

	return poweredLuminance * poweredContrast * poweredStructure; // ssim formula
}

double average(Mat& img) {
	double result = 0.0;
	int imgSize = img.rows * img.cols; 

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);

			int r = pixel[2];
			int g = pixel[1];
			int b = pixel[0];

			result += (double)(r + g + b) / 3.0; // avg color (simple gray scale)
		}
	}
	result /= imgSize; // avg for all image

	return result;
}

double variance(Mat& img, double muX) {
	double result = 0.0;
	int imgSize = img.rows * img.cols;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			
			int r = pixel[2];
			int g = pixel[1];
			int b = pixel[0];
			
			double shiftedValue = ( r + g + b ) / 3.0 - muX; // shifted avg color(simple gray scale)
			
			result += shiftedValue * shiftedValue; // squre of shifted value. needed for formula
		}
	}

	result = sqrt( result / ( imgSize - 1 ) ); // imgSize - 1 = N - 1

	return result;
}

double covariance(Mat& img1, Mat& img2, double muX, double muY) {
	int imgSize = img1.rows * img1.cols;
	double result = 0.0;

	//its needed because we cant calculate covariance for vectors of different dimensions
	if ( (img1.rows != img2.rows) && (img1.cols != img2.cols) ) {
		throw "image sizes aren't equal";
	}

	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			Vec3b pixelX = img1.at<Vec3b>(i, j);
			Vec3b pixelY = img2.at<Vec3b>(i, j);

			int rX = pixelX[2];
			int gX = pixelX[1];
			int bX = pixelX[0];

			double shiftedValueX = (rX + gX + bX) / 3.0 - muX; // shifted avg color( simple gray scale) for first img(vector)

			int rY = pixelY[2];
			int gY = pixelY[1];
			int bY = pixelY[0];

			double shiftedValueY = (rY + gY + bY) / 3.0 - muY; // shifted avg color( simple gray scale) for second img(vector)

			result += shiftedValueX * shiftedValueY; // squre, needed for formula of covariance
		}
	}
	result /= imgSize; // normalize

	return result;
}

void calculateAndPrintParametres(Mat& img1, Mat& img2) {
	double _averageX = 0.0, _averageY = 0.0,
		_varianceX = 0.0, _varianceY = 0.0,
		_covarianceXY = 0.0, _luminanceXY = 0.0,
		_contrastXY = 0.0, _structureXY = 0.0,
		_ssim = 0.0;

	// calculating parametres for first image
	_averageX = average(img1);
	_varianceX = variance(img1, _averageX);

	//calculating parametres for second image
	_averageY = average(img2);
	_varianceY = variance(img2, _averageY);

	//calculating covariance
	_covarianceXY = covariance(img1, img2, _averageX, _averageY);

	//calculating luminance, contrast, structure
	_luminanceXY = luminance(_averageX, _averageY);
	_contrastXY = contrast(_varianceX, _varianceY);
	_structureXY = structure(_covarianceXY, _varianceX, _varianceY);

	//calculating ssim
	_ssim = ssim(_luminanceXY, _contrastXY, _structureXY);

	//printing parametres for first image
	cout << "***First image***" << endl << " **Average: " << _averageX << " **" << endl << " **Variance: " << _varianceX << " **" << endl << endl;

	//printing parametres for second image
	cout << "***Second image***" << endl << " **Average: " << _averageY << " **" << endl << " **Variance: " << _varianceY << " **" << endl << endl;

	//printing covariance, luminance, contrast, structure
	cout << "***Comparative parameters for two images***" << endl
		<< " **Covariance: " << _covarianceXY << " **" << endl
		<< " **Luminance: " << _luminanceXY << " **" << endl
		<< " **Contrast: " << _contrastXY << " **" << endl
		<< " **Structure: " << _structureXY << " **" << endl << endl << endl;
	
	//printing resulting ssim
	cout << "***Result***" << endl << " **SSIM: " << _ssim << " **" << endl << endl << endl << endl << endl;

}