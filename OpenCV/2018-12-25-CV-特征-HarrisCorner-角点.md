# Harris Corner 角点


[本文转载出处: http://www.cnblogs.com/ronny/p/4009425.html](http://www.cnblogs.com/ronny/p/4009425.html)

特征点选取一般要具有重复性、可区分性、数量适宜、有效性等特点。角点与直线、平面相比，更具有可区分性。

Harris Conner 是最典型的角点检测子。


<!-- 插入图片的方法 ![2018-12-25-CV-特征-Harris Corner 角点](2018-12-25-CV-特征-Harris Corner 角点/test.jpg) --> 

## 1. 不同类型的角点

在现实世界中，角点对应于物体的拐角，道路的十字路口、丁字路口等。从图像分析的角度来定义角点可以有以下两种定义：

1. 角点可以是两个边缘的角点；
2. 角点是邻域内具有两个主方向的特征点；

前者往往需要对图像边缘进行编码，这在很大程度上依赖于图像的分割与边缘提取，具有相当大的难度和计算量，且一旦待检测目标局部发生变化，很可能导致操作的失败。早期主要有Rosenfeld和Freeman等人的方法，后期有CSS等方法。

基于图像灰度的方法通过计算点的曲率及梯度来检测角点，避免了第一类方法存在的缺陷，此类方法主要有Moravec算子、Forstner算子、Harris算子、SUSAN算子等。

![](corner_history.jpg)

这篇文章主要介绍的Harris角点检测的算法原理，比较著名的角点检测方法还有jianbo Shi和Carlo Tomasi提出的Shi-Tomasi算法，这个算法开始主要是为了解决跟踪问题，用来衡量两幅图像的相似度，我们也可以把它看为Harris算法的改进。OpenCV中已经对它进行了实现，接口函数名为[GoodFeaturesToTrack()](http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack)。另外还有一个著名的角点检测算子即SUSAN算子，SUSAN是Smallest Univalue Segment Assimilating Nucleus（最小核值相似区）的缩写。SUSAN使用一个圆形模板和一个圆的中心点，通过圆中心点像素与模板圆内其他像素值的比较，统计出与圆中心像素近似的像元数量，当这样的像元数量小于某一个阈值时，就被认为是要检测的角点。我觉得可以把SUSAN算子看为Harris算法的一个简化。这个算法原理非常简单，算法效率也高，所以在OpenCV中，它的接口函数名称为：[FAST()](https://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html#fast) 。

## 2. Harris Conner

### 2.1 基本原理

人眼对角点的识别通常是在一个局部的小区域或小窗口完成的。如果在各个方向上移动这个特征的小窗口，窗口内区域的灰度发生了较大的变化，那么就认为在窗口内遇到了角点。如果这个特定的窗口在图像各个方向上移动时，窗口内图像的灰度没有发生变化，那么窗口内就不存在角点；如果窗口在某一个方向移动时，窗口内图像的灰度发生了较大的变化，而在另一些方向上没有发生变化，那么，窗口内的图像可能就是一条直线的线段。

![](plane_line_corner.png)

对于图像 $I(x,y)$ ，当在点 $(x,y)$ 出平移了 $(\Delta x,\Delta y)$ 后的自相关性，可通过自相关函数给出：

$$
c(x,y,\Delta x,\Delta y)=\sum_{(u,v)\in W(x,y)}\omega(u,v)(I(u,v)-I(u+\Delta x,v+\Delta y))^2
$$

其中， $W(x,y)$ 是以点 $(x,y)$ 为中点的窗口， $\omega(u,v)$ 是加权函数，既可以是常数，也可以是高斯加权函数。

根据泰勒展开，对图像 $I(x,y)$ 在平移 $(\Delta x,\Delta y)$ 后进行一阶近似：

$$
I(u+\Delta x,v+\Delta y)=I(u,v)+I_x(u,v)\Delta x+I_y(u,v)\Delta y+O(\Delta x^2,\Delta y^2)\\
\approx I(u,v)+I_x(u,v)\Delta x+I_y(u,v)\Delta y
$$

其中， $I_x,I_y$ 是图像 $I(x,y)$  的偏导数，这样的话，自相关函数则可以简化为：
$$
c(x,y,\Delta x,\Delta y)\approx \sum_\omega (I_x(u,v)\Delta x+I_y(u,v)\Delta y)^2\\
=[\begin{array}{c}\Delta x&\Delta y\end{array}]M(x,y)\left[\begin{array}{c}\Delta x\\ \Delta y\end{array}\right]
$$

其中：
$$
M(x,y)=\sum_\omega\left[\begin{array}{cc}I_x(x,y)^2&I_x(x,y)I_y(x,y)\\I_x(x,y)I_y(x,y)&I_y(x,y)^2\end{array}\right]\\
=\left[\begin{array}{cc}\sum_\omega I_x(x,y)^2&\sum_\omega I_x(x,y)I_y(x,y)\\\sum_\omega I_x(x,y)I_y(x,y)&\sum_\omega I_y(x,y)^2\end{array}\right]\\
=\left[\begin{array}{cc}A&C\\C&B\end{array}\right]
$$

图像 $I(x,y)$ 在点 $(x,y)$ 处平移 $(\Delta x,\Delta y)$ 后的自相关函数可以近似为二次项函数：
$$
c(c,y,\Delta x,\Delta y)\approx A\Delta x^2+2C\Delta x\Delta y+B\Delta y^2
$$
其中
$$
A=\sum_\omega I_x^2,B=\sum_\omega I_y^2,C=\sum_\omega I_xI_y
$$

二次项函数本质上是一个椭圆函数。椭圆的扁率和尺寸是由 $M(x,y)$ 的特征值 $\lambda_1,\lambda_2$ 决定的。椭圆的方向由 $M(x,y)$ 的特征矢量决定，如下图所示，椭圆方程为：

$$
\left[\begin{array}{c}\Delta x&\Delta y\end{array}\right]M(x,y)\left[\begin{array}{cc}\Delta x\\\Delta y\end{array}\right]=1
$$

![](ellipse.png)

椭圆函数特征值与图像中的角点、直线（边缘）和平面之间的关系如下图所示。共可分为三种情况：
- 图像中的直线。一个特征值大，另一个特征值小， $\lambda_1\gg\lambda_2$ 或 $\lambda_2\gg\lambda_1$ 。自相关函数值在某一方向上大，在其他方向上小。
- 图像中的平面。两个特征值都小，且近似相等；自相关函数数值在各个方向上都小。
- 图像中的角点。两个特征值都大，且近似相等，自相关函数在所有方向都增大。

根据二次项函数特征值的计算公式，我们可以求 $M(x,y)$ 矩阵的特征值。但是 Harris 给出的角点差别方法并不需要计算具体的特征值，而是计算一个角点响应值 $R$ 来判断角点。 $R$ 的计算公式为：
$$
R=det\mathbf M-\alpha(trace\mathbf M)^2
$$

式中， $det\mathbf M$ 为矩阵 $M=\left[\begin{array}{cc}A&C\\C&B\end{array}\right]$ 的行列式；$trace\mathbf M$ 为 矩阵 $\mathbf M$ 的迹； $\alpha$ 为常数，取值范围一般为 $0.04\sim0.06$，是为了抑制比较明显的直线。事实上特征是隐含在 $det\mathbf M$ 和 $trace\mathbf M$ 中的，因为：
$$
det\mathbf M=\lambda_1\lambda_2=AB-C^2\\
trace\mathbf M=\lambda_1+\lambda_2=A+B
$$


### 2.2 Harris Conner 算法实现

根据上述讨论，Harris Conner 检测算法可分为以下5步：
1. 计算图像 $I(x,y)$ 在 $X,Y$ 两个方向上的梯度 $I_x,I_y$。
$$
I_x=\dfrac{\partial I}{\partial x}=I\otimes(-1,0,1),I_y=\dfrac{\partial I}{\partial y}=I\otimes(-1,0,1)^T
$$
2. 计算图像两个方向梯度的乘积。
$$
I_x^2=I_x\cdot I_x,I_y^2=I_y\cdot I_y,I_{xy}=I_x\cdot I_y
$$

3. 使用高斯加权对 $I_x^2,I_y^2,I_{xy}$ 进行高斯加权（取 $\sigma=1$ ），生成矩阵 $M$ 的元素 $A,B,C$ 。
$$
A=g(I_x^2)=I_x^2\otimes\omega,B=g(I_y^2)=I_y^2\otimes\omega,C=g(I_{xy})=I_{xy}\otimes\omega,
$$

4. 计算每个像素的 Harris 响应值 $R$ ，并对于小于某一阈值 $t$ 的 $R$ 置为零。

$$
R=\left\{R:det\mathbf M-\alpha(trace\mathbf M)^2<t\right\}
$$

5. 在3x3或5x5的领域内进行[非极大值抑制（NMS）](https://baike.baidu.com/item/%E9%9D%9E%E6%9E%81%E5%A4%A7%E5%80%BC%E6%8A%91%E5%88%B6)，局部最大值点即为图像中的角点。

### 2.3 OpenCV 代码

代码：

``` cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
	Mat srcImage;	// 原图
	Mat dstImage;	// 目标图
	Mat normImage;	// 归一化图
	Mat scaledImage;// 线性变换后八位无符号整型图

	// 打开图片
	srcImage = imread("img.jpg", 0);
	// 显示原图
	imshow("原图", srcImage);
	imwrite("img_gray.jpg", srcImage);

	dstImage = Mat::zeros(srcImage.size(), CV_32FC1);
	// 角点检测
	cornerHarris(srcImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);
	// 归一化处理
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	// 转换为8位无符号整型
	scaledImage = Mat(srcImage.size(), CV_8UC1);
	convertScaleAbs(normImage, scaledImage);
	
	imshow("corner", scaledImage);
	imwrite("img_corner.jpg", scaledImage);

	waitKey(0);

	return 1;
}
```


### 2.4 自编代码实现 Harris Corner

代码：

``` cpp
void mHarrisCorner(Mat& srcImage, Mat& dstImage, double alpha)
{
	Mat gray;

	gray = srcImage.clone();
	gray.convertTo(gray, CV_64F);

	Mat xKernel = (Mat_<double>(1, 3) << -1, 0, 1);
	Mat yKernel = xKernel.t();

	Mat Ix, Iy;
	filter2D(gray, Ix, CV_64F, xKernel);
	filter2D(gray, Iy, CV_64F, yKernel);

	Mat Ix2, Iy2, Ixy;
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);

	Mat gaussKernel = getGaussianKernel(5, 1);
	filter2D(Ix2, Ix2, CV_64F, gaussKernel);
	filter2D(Iy2, Iy2, CV_64F, gaussKernel);
	filter2D(Ixy, Ixy, CV_64F, gaussKernel);

	Mat cornerStrength(gray.size(), gray.type());
	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			double det_m = Ix2.at<double>(i, j) * Iy2.at<double>(i, j) - Ixy.at<double>(i, j) * Ixy.at<double>(i, j);
			double trace_m = Ix2.at<double>(i, j) + Iy2.at<double>(i, j);
			cornerStrength.at<double>(i, j) = det_m - alpha * trace_m *trace_m;
		}
	}

	dstImage = cornerStrength.clone();
}
```

### 2.5 Harris 角点的性质

#### 2.5.1 参数 $\alpha$ 对的影响

假设 $\lambda_1\geqslant\lambda_2\geqslant0$ ，令 $\lambda_2=k\lambda_1, 0\leqslant k\leqslant1$ 。从而有：
$$
R=\lambda_1\lambda_2-\alpha(\lambda_1+\lambda_2)^2=\lambda_1^2(k-\alpha(1+k)^2)
$$

假设 $R\geqslant0$，则有：
$$
0\leqslant\alpha\leqslant\dfrac{k}{(1+k)^2}\leqslant0.25
$$

由此，可以得出这样的结论：增大 $\alpha$ 的值，将减小角点响应值 $R$ ，降低角点检测的灵敏性，减少被检测角点的数量；减小$\alpha$ 值，将增大角点响应值 $R$ ，增加角点检测的灵敏性，增加被检测角点的数量。

#### 2.5.2 对亮度和对比度的变化不敏感

这是因为在进行 Harris 角点检测时，使用了微分算子对图像进行微分运算，而微分运算对图像密度的拉升或收缩和对亮度的抬高或下降不敏感。换言之，对亮度和对比度的仿射变换并不改变 Harris 响应的极值点出现的位置，但是，由于阈值的选择，可能会影响角点检测的数量。

#### 2.5.3 具有旋转不变性

Harris 角点检测算子使用的是角点附近的区域灰度二阶矩矩阵。而二阶矩矩阵可以表示成一个椭圆，椭圆的长短轴正是二阶矩矩阵特征值平方根的倒数。当特征椭圆转动时，特征值并不发生变化，所以判断角点响应值 $R$ 也不发生变化，由此说明 Harris 角点检测算子具有旋转不变性。

#### 2.5.4 不具备尺度不变性

如下图所示，当右图被缩小时，在检测窗口尺寸不变的前提下，在窗口内所包含图像的内容是完全不同的。左侧的图像可能被检测为边缘或曲线，而右侧的图像则可能被检测为一个角点。


## 3 多尺度 Harris 角点

尺度理论参考：http://www.cnblogs.com/ronny/p/3886013.html

### 3.1 多尺度Harris角点原理

虽然 Harris 角点检测算子具有部分图像灰度变化的不变性和旋转不变性，但它不具有尺度不变性。但是尺度不变性对图像特征来说至关重要。人们在使用肉眼识别物体时，不管物体远近，尺寸的变化都能认识物体，这是因为人的眼睛在辨识物体时具有较强的尺度不变性。

在上面给出的连接中介绍了高斯尺度空间的概念。下面将Harris角点检测算子与高斯尺度空间表示相结合，使用Harris角点检测算子具有尺度的不变性。

仿照 Harris 角点检测中二阶矩的表示方法，使用$M=\mu(x,\sigma _I,\sigma_D)$为尺度自适应的二阶矩：

$$
M=\mu(x,\sigma_I,\sigma_D)=\sigma_D^2g(\sigma_I)\otimes\left[\begin{array}{cc}L_x^2(x,\sigma_D)&L_xL_y(x,\sigma_D)\\L_xL_y(x,\sigma_D)&L_y^2(x,\sigma_D)\end{array}\right]
$$

其中，$g(\sigma_I)$ 表示尺度为 $\sigma_I$ 的高斯卷积核， $x$ 表示图像的位置。与高斯测度空间类似，使用 $L(x)$ 表示经过高斯平滑后的图像，符号 $\otimes$ 表示卷积， $L_x(x,\sigma_D),L_y(x,\sigma_D)$ 表示对图像使用高斯 $g(\sigma_D)$ 函数进行平滑后，在 $x,y$ 方向取其微分的结果，即 $L_x=\partial_xL,L_y=\partial_yL$ 。通常将 $\sigma_I$ 称为积分尺度，它是决定 Harris 角点当前尺寸的变量， $\sigma_D$ 是微分尺度或局部尺度，它是决定角点附近微分值得变量。显然，积分尺度 $\sigma_I$ 应该大于微分尺度 $\sigma_D$ 。

### 2.2 多尺度 Harris 角点实现

首先，检测算法从预先定义的一组尺度中进行积分尺度搜索，这一组尺度定义为
$$
\sigma_1...\sigma_n=\sigma_0...k^n\sigma_0
$$

一般情况下使用 $k=1.4$ 。为了减少搜索的复杂性，对于微分尺度 $\sigma_D$ 的选择，我们采用在积分尺度的基础上，乘以一个比例常数，即 $\sigma_D=s\sigma_I$ ，一般取 $s=0.7$ 。这样，通常使用积分和微分的尺度，便可以生成 $\mu(x,\sigma_I,\sigma_D)$ ，再利用 Harris 角点判断准则，对角点进行搜索，具体可以分为两步进行。

1. 与 Harris 角点搜索类似，对于给定的尺度空间 $\sigma_D$ ，进行如下角点响应值计算和判断：
$$
cornerness=det(\mu(x,\sigma_n))-\alpha trace^2(\mu(x,\sigma_n)) > threshold_H
$$

2. 对于满足1中条件的点，在点的8邻域内进行角点响应最大值搜索（即非最大值抑制）出在8邻域内角点响应最大值的点。对于每个尺度 $\sigma(1,2,...,n)$ 都进行如上搜索。

由于位置空间的候选点并不一定在尺度空间上也能成为候选点，所以，我们还要在尺度空间上进行搜索，找到该点的所谓特征尺度值。搜索特征尺度值也分两步。

1. 对于位置空间搜索到的每个候选点，进行拉普拉斯响应计算，并满足其绝对值大于给定的阈值条件：
$$
F(x,\sigma_n)=\sigma_n^2|L_{xx}(x,\sigma_n)+L_{yy}(x,\sigma_n)|\geqslant threshold_L
$$

2. 与邻近的两个尺度空间的拉普拉斯响应值进行比较，使其满足：
$$
F(x,\sigma_n)>F(x,\sigma_l), l\in\{n-1,n+1\}
$$

满足上述条件的尺度值就是改点的特征尺度值，这样，我们就找到了在位置空间和尺度空间都满足条件的 Harris 角点。

### 2.3 多尺度Harris编程实现

代码：
```cpp
todo
```

效果：
todo


## 3 Shi-Tomasi 算法

Shi-Tomasi 算法是Harris 算法的改进。Harris 算法最原始的定义是将矩阵 $M$ 的行列式值与 $M$ 的迹相减，再将差值同预先给定的阈值进行比较。后来Shi 和Tomasi 提出改进的方法，若两个特征值中较小的一个大于最小阈值，则会得到强角点。

对自相关矩阵 M 进行特征值分析，产生两个特征值 $(\lambda_1,\lambda_2)$ 和两个特征方向向量。因为较大的不确定度取决于较小的特征值，也就是 $a_0^{-\frac{1}{2}}$ ，所以通过寻找最小特征值的最大值来寻找好的特征点也就解释的通了。

Shi-Tomasi 的方法比较充分，并且在很多情况下可以得到比使用 Harris 算法更好的结果。

Opencv 中 Shi-Tomasi 的实现函数为 goodFeaturesToTrack 。

### 3.1 OpenCv 实现

```cpp
//--------------------------------------【程序说明】-------------------------------------------
//		程序说明：《OpenCV3编程入门》OpenCV3版书本配套示例程序87
//		程序描述：Shi-Tomasi角点检测示例
//		开发测试所用操作系统： Windows 7 64bit
//		开发测试所用IDE版本：Visual Studio 2010
//		开发测试所用OpenCV版本：	3.0 beta
//		2014年11月 Created by @浅墨_毛星云
//		2014年12月 Revised by @浅墨_毛星云
//------------------------------------------------------------------------------------------------



//---------------------------------【头文件、命名空间包含部分】----------------------------
//		描述：包含程序所使用的头文件和命名空间
//------------------------------------------------------------------------------------------------
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//----------------------------------------------------------------------------------------------
#define WINDOW_NAME "【Shi-Tomasi角点检测】"        //为窗口标题定义的宏 



//-----------------------------------【全局变量声明部分】--------------------------------------
//          描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_grayImage;
int g_maxCornerNumber = 33;
int g_maxTrackbarNumber = 500;
RNG g_rng(12345);//初始化随机数生成器


				 //-----------------------------【on_GoodFeaturesToTrack( )函数】----------------------------
				 //          描述：响应滑动条移动消息的回调函数
				 //----------------------------------------------------------------------------------------------
void on_GoodFeaturesToTrack(int, void*)
{
	//【1】对变量小于等于1时的处理
	if (g_maxCornerNumber <= 1) { g_maxCornerNumber = 1; }

	//【2】Shi-Tomasi算法（goodFeaturesToTrack函数）的参数准备
	vector<Point2f> corners;
	double qualityLevel = 0.01;//角点检测可接受的最小特征值
	double minDistance = 10;//角点之间的最小距离
	int blockSize = 3;//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;//权重系数
	Mat copy = g_srcImage.clone();	//复制源图像到一个临时变量中，作为感兴趣区域

									//【3】进行Shi-Tomasi角点检测
	goodFeaturesToTrack(g_grayImage,//输入图像
		corners,//检测到的角点的输出向量
		g_maxCornerNumber,//角点的最大数量
		qualityLevel,//角点检测可接受的最小特征值
		minDistance,//角点之间的最小距离
		Mat(),//感兴趣区域
		blockSize,//计算导数自相关矩阵时指定的邻域范围
		false,//不使用Harris角点检测
		k);//权重系数


		   //【4】输出文字信息
	cout << "\t>此次检测到的角点数量为：" << corners.size() << endl;

	//【5】绘制检测到的角点
	int r = 4;
	for (int i = 0; i < corners.size(); i++)
	{
		//以随机的颜色绘制出角点
		circle(copy, corners[i], r, Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255),
			g_rng.uniform(0, 255)), -1, 8, 0);
	}

	//【6】显示（更新）窗口
	imshow(WINDOW_NAME, copy);
}


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第87个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
	printf("\n\n  ----------------------------------------------------------------------------\n");
	//输出一些帮助信息
	printf("\n\n\n\t欢迎来到【Shi-Tomasi角点检测】示例程序\n");
	printf("\n\t请调整滑动条观察图像效果\n\n");

}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	//【0】改变console字体颜色
	system("color 2F");

	//【0】显示帮助文字
	ShowHelpText();

	//【1】载入源图像并将其转换为灰度图
	g_srcImage = imread("1.jpg", 1);
	cvtColor(g_srcImage, g_grayImage, COLOR_RGB2GRAY);

	//【2】创建窗口和滑动条，并进行显示和回调函数初始化
	namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
	createTrackbar("最大角点数", WINDOW_NAME, &g_maxCornerNumber, g_maxTrackbarNumber, on_GoodFeaturesToTrack);
	imshow(WINDOW_NAME, g_srcImage);
	on_GoodFeaturesToTrack(0, 0);

	waitKey(0);
	return(0);
}
```



### 3.2 自己实现

代码：

```cpp
void badFeaturesToTrack(Mat & img, Mat & corners)
{
	Mat gray;
	gray = img.clone();
	gray.convertTo(gray, CV_64F);

	Mat xKernel = (Mat_<double>(1, 3) << -1, 0, 1);
	Mat yKernel = xKernel.t();

	Mat Ix, Iy;
	filter2D(gray, Ix, CV_64F, xKernel);
	filter2D(gray, Iy, CV_64F, yKernel);

	Mat Ix2, Iy2, Ixy;
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);

	/*Mat gaussKernel = getGaussianKernel(5, 1);
	filter2D(Ix2, Ix2, CV_64F, gaussKernel);
	filter2D(Iy2, Iy2, CV_64F, gaussKernel);
	filter2D(Ixy, Ixy, CV_64F, gaussKernel);*/

	boxFilter(Ix2, Ix2, CV_64F, Size(3,3));
	boxFilter(Ixy, Ixy, CV_64F, Size(3, 3));
	boxFilter(Iy2, Iy2, CV_64F, Size(3, 3));

	Size size = gray.size();

	Mat dst(size, gray.type());
	for (int i = 0; i < size.height; i++)
	{
		double *dst_data = dst.ptr<double>(i);
		double *ix2 = Ix2.ptr<double>(i);
		double *ixy = Ixy.ptr<double>(i);
		double *iy2 = Iy2.ptr<double>(i);
		for (int j = 0; j < size.width; j++)
		{
			double a = 0.5*ix2[j];
			double b = ixy[j];
			double c = 0.5*iy2[j];
			dst_data[j] = (double)((a + c) - std::sqrt((a - c)*(a - c) + b*b));
		}
	}

	corners = dst.clone();
}
```

## 总结

OpenCV 实现的 goodFeaturesToTrack 效果最好。除了Shi-Tomasi 的实现，还有特征点最小间隔、特征点排序等功能的实现。
