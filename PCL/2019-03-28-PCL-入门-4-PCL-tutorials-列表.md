# PCL 入门-3-PCL tutorials 列表


这里对 [PCL 官方 tutorials](http://www.pointclouds.org/documentation/tutorials/) 做个简单的分析理解，便于后面应用时查找相应的例子。



## 1. 基本使用

- 1.1 [PCL Walkthrough](http://www.pointclouds.org/documentation/tutorials/walkthrough.php#walkthrough)

大概了解 PCL 的基本构成.

- 1.2 [Getting Started / Basic Structures](http://www.pointclouds.org/documentation/tutorials/basic_structures.php#basic-structures)

点云数据类型 pcl::PointCloud 的结构。

- 1.3 [Using PCL in your own project](http://www.pointclouds.org/documentation/tutorials/using_pcl_pcl_config.php#using-pcl-pcl-config)

利用 CMake 进行自己的项目的构建。

- 1.4 [Compiling PCL from source on POSIX compliant systems](http://www.pointclouds.org/documentation/tutorials/compiling_pcl_posix.php#compiling-pcl-posix)

POSIX 系统上编译 PCL。

- 1.5 [Customizing the PCL build process](http://www.pointclouds.org/documentation/tutorials/building_pcl.php#building-pcl)

自定义 PCL 编译选项。

- 1.6 [Building PCL’s dependencies from source on Windows](http://www.pointclouds.org/documentation/tutorials/compiling_pcl_dependencies_windows.php#compiling-pcl-dependencies-windows)

在 Windows 上编译 PCL 依赖。

- 1.7 [Compiling PCL from source on Windows](http://www.pointclouds.org/documentation/tutorials/compiling_pcl_windows.php#compiling-pcl-windows)

Windows 上源码编译 PCL。

- 1.8 [Compiling PCL and its dependencies from MacPorts and source on Mac OS X](http://www.pointclouds.org/documentation/tutorials/compiling_pcl_macosx.php#compiling-pcl-macosx)

Mac OS X 上编译 PCL 及其依赖。

- 1.9 [Installing on Mac OS X using Homebrew](http://www.pointclouds.org/documentation/tutorials/installing_homebrew.php#installing-homebrew)

Mac OS X 上用 Homebrew 安装 PCL。

- 1.10 [Using PCL with Eclipse](http://www.pointclouds.org/documentation/tutorials/using_pcl_with_eclipse.php#using-pcl-with-eclipse)

用 Eclipse 进行 PCL 开发。

- 1.11 [Generate a local documentation for PCL](http://www.pointclouds.org/documentation/tutorials/generate_local_doc.php#generate-local-doc)

生成本地参考文档。

- 1.12 [Using a matrix to transform a point cloud](http://www.pointclouds.org/documentation/tutorials/matrix_transform.php#matrix-transform)

例子，读取 PCD 点云文件，然后使用一个矩阵对点云进行操作，并用 pcl::visualization::PCLVisualizer 进行可视化。

## 2. 高级使用方法

- 2.1 [Adding your own custom PointT type](http://www.pointclouds.org/documentation/tutorials/adding_custom_ptype.php#adding-custom-ptype)

添加使用自定义的点类型 PointT。

- 2.2 [Writing a new PCL class](http://www.pointclouds.org/documentation/tutorials/writing_new_classes.php#writing-new-classes)

自定义 PCL class。

## 3. 特征 Features

- 3.1 [How 3D Features work in PCL](http://www.pointclouds.org/documentation/tutorials/how_features_work.php#how-3d-features-work)

介绍 PCL 中的特征是怎么一回事，相关理论。

良好的特征描述需要的特点：

- 在刚体变换中，特征描述不变
- 在不同采样密度下，特征描述不变
- 可以抗噪声干扰

一般来说，PCL 中关注两种邻域点搜索方式：

- 确定 k 个邻域点（k-search）
- 确定在半径 r 内的邻域点（radius-search）

搜索到邻域点后，再利用这些邻域点进行特征描述运算。

- 3.2 [Estimating Surface Normals in a PointCloud](http://www.pointclouds.org/documentation/tutorials/normal_estimation.php#normal-estimation)

点云面法线估计。

法线估计的方法有很多，这里介绍最小二乘平面拟合方法。

然后讲了如何选择法线的符号，即朝向问题。

因为法线估计需要用到 k-neighborhood 的点，所以 k 或 r 的取值大小（scale）需要合理选择。

法线估计可以使用 **OpenMP** 进行多线程加速。

如果点云数据是有序的，那么可以使用下面的积分图加速算法。

- 3.3 [Normal Estimation Using Integral Images](http://www.pointclouds.org/documentation/tutorials/normal_estimation_using_integral_images.php#normal-estimation-using-integral-images)

利用**积分图**对**有序点云**进行面法线估计。

- 3.4 [Point Feature Histograms (PFH) descriptors](http://www.pointclouds.org/documentation/tutorials/pfh_estimation.php#pfh-estimation)

PFH 特征描述需要使用 XYZ 坐标信息以及 面法向信息。

*computePairFeatures* 将两个点的坐标信息xyz，和 normal 信息共 12 个值转为 4 个值来描述两个点之间的关系。

```cpp
computePairFeatures (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1,
                     const Eigen::Vector4f &p2, const Eigen::Vector4f &n2,
                     float &f1, float &f2, float &f3, float &f4);
```

然后根据4个描述值，对 k-neighborhood 内的进行直方图统计。

- 3.5 [Fast Point Feature Histograms (FPFH) descriptors](http://www.pointclouds.org/documentation/tutorials/fpfh_estimation.php#fpfh-estimation)

PFH 算法的时间复杂度是 $O(nk^2)$, FPFH 算法的时间复杂度可以减少到 $O(nk)$。

- 3.6 [Estimating VFH signatures for a set of points](http://www.pointclouds.org/documentation/tutorials/vfh_estimation.php#vfh-estimation)

Viewpoint Feature Histogram (VFH) 视点特征直方图 是 一种新型的对点聚类的表示方法，用于**聚类识别**和**6DOF姿态识别**。

- 3.7 [How to extract NARF Features from a range image](http://www.pointclouds.org/documentation/tutorials/narf_feature_extraction.php#narf-feature-extraction)

Normal Aligned Radial Feature (NARF) 特征，是针对 **Range Image** 的特征，是一种二维图像的特征。

- 3.8 [Moment of inertia and eccentricity based descriptors](http://www.pointclouds.org/documentation/tutorials/moment_of_inertia.php#moment-of-inertia)

*pcl::MomentOfInertiaEstimation* 来获取 *descriptors based on eccentricity and moment of inertia*（基于偏心和惯性矩的特征描述）。

- 3.9 [RoPs (Rotational Projection Statistics) feature](http://www.pointclouds.org/documentation/tutorials/rops_feature.php#rops-feature)

RoPs (Rotational Projection Statistics) feature. 旋转投影统计特征。

- 3.10 [Globally Aligned Spatial Distribution (GASD) descriptors](http://www.pointclouds.org/documentation/tutorials/gasd_estimation.php#gasd-estimation)

全局对齐空间分布描述特征 Globally Aligned Spatial Distribution (GASD) descriptors 主要用于物体识别（object pose）以及姿态估计。

首先根据所有点的信息，先将它与标准坐标系进行对齐，然后根据对齐的点的空间分布求出特征描述。

## 4. 滤波 Filtering

- 4.1 [Filtering a PointCloud using a PassThrough filter](http://www.pointclouds.org/documentation/tutorials/passthrough.php#passthrough)

用于去除特定轴上特定范围内或外的点。

- 4.2 [Downsampling a PointCloud using a VoxelGrid filter](http://www.pointclouds.org/documentation/tutorials/voxel_grid.php#voxelgrid)

用体素格 VoxelGrid 进行下采样。

- 4.3 [Removing outliers using a StatisticalOutlierRemoval filter](http://www.pointclouds.org/documentation/tutorials/statistical_outlier.php#statistical-outlier-removal)

对每一个点，统计其与邻近点(Mean-K)的距离，如果它的距离与统计出来的正态平均值差距较大，则认为是离群点。

- 4.4 [Projecting points using a parametric model](http://www.pointclouds.org/documentation/tutorials/project_inliers.php#project-inliers)

投影点到参数化模型上，如平面。

- 4.5 [Extracting indices from a PointCloud](http://www.pointclouds.org/documentation/tutorials/extract_indices.php#extract-indices)

使用 pcl::ExractIndices 函数基于 Segmenation 算法求出的 indices 进行点云提取。

- 4.6 [Removing outliers using a Conditional or RadiusOutlier removal](http://www.pointclouds.org/documentation/tutorials/remove_outliers.php#remove-outliers)

使用 RadiusOutlierRemoval，可以设置 在半径 r 的范围内，如果少于 n 个邻近点，则去除当前点。

使用 ConditionalRemoval ， 可以设置 根据属性值如 z 的大小满足设置的条件是进行过滤。

## 5. IO

- 5.1 [The PCD (Point Cloud Data) file format](http://www.pointclouds.org/documentation/tutorials/pcd_file_format.php#pcd-file-format)

介绍 PCD 格式，特点。

- 5.2 [Reading Point Cloud data from PCD files](http://www.pointclouds.org/documentation/tutorials/reading_pcd.php#reading-pcd)

读取 PCD 文件。

- 5.3 [Writing Point Cloud data to PCD files](http://www.pointclouds.org/documentation/tutorials/writing_pcd.php#writing-pcd)

点云数据写入 PCD 文件。

- 5.4 [Concatenate the points of two Point Clouds](http://www.pointclouds.org/documentation/tutorials/concatenate_clouds.php#concatenate-clouds)

两个点云连接为一个点云。

- 5.5 [The OpenNI Grabber Framework in PCL][http://www.pointclouds.org/documentation/tutorials/openni_grabber.php#openni-grabber]

使用 OpenNI 获取点云数据。

- 5.6 [The Velodyne High Definition LiDAR (HDL) Grabber](http://www.pointclouds.org/documentation/tutorials/hdl_grabber.php#hdl-grabber)

- 5.7 [The PCL Dinast Grabber Framework](http://www.pointclouds.org/documentation/tutorials/dinast_grabber.php#dinast-grabber)

- 5.8 [Grabbing point clouds from Ensenso cameras](http://www.pointclouds.org/documentation/tutorials/ensenso_cameras.php#ensenso-cameras)

- 5.9 [Grabbing point clouds / meshes from davidSDK scanners](http://www.pointclouds.org/documentation/tutorials/davidsdk.php#david-sdk)

- 5.10 [Grabbing point clouds from DepthSense cameras](http://www.pointclouds.org/documentation/tutorials/depth_sense_grabber.php#depth-sense-grabber)


## 6. Keypoints

- 6.1 [How to extract NARF keypoint from a range image](http://www.pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction)

## 7. KdTree

- 7.1 [How to use a KdTree to search](http://www.pointclouds.org/documentation/tutorials/kdtree_search.php#kdtree-search)

KdTree 中 Kd 是指 k-dimenional，是一种分割 k 维数据空间的数据结构，主要用于多为空间关键数据的搜索，如范围搜索和最近邻搜索。

## 8. OcTree

- 8.1 [Point Cloud Compression](http://www.pointclouds.org/documentation/tutorials/compression.php#octree-compression)

利用 OcTree（八叉树）进行点云的压缩，便于传输。

- 8.2 [Spatial Partitioning and Search Operations with Octrees](http://www.pointclouds.org/documentation/tutorials/octree.php#octree-search)

利用 OcTree 进行空间分区和搜索操作。

- 8.3 [Spatial change detection on unorganized point cloud data](http://www.pointclouds.org/documentation/tutorials/octree_change.php#octree-change-detection)

利用 OcTree 进行空间点云变换的检测，如某个空间本来没有点，新的点云又有了点。

## 9. Range Images

- 9.1 [How to create a range image from a point cloud](How to create a range image from a point cloud)

用点云生产深度图像。

- 9.2 [How to extract borders from range images](http://www.pointclouds.org/documentation/tutorials/range_image_border_extraction.php#range-image-border-extraction)

在深度图像上提取边界。


## 10. Recognition

- 10.1 [3D Object Recognition based on Correspondence Grouping](http://www.pointclouds.org/documentation/tutorials/correspondence_grouping.php#correspondence-grouping)

..

- 10.2 [Implicit Shape Model](http://www.pointclouds.org/documentation/tutorials/implicit_shape_model.php#implicit-shape-model)

..

- 10.3 [Tutorial: Hypothesis Verification for 3D Object Recognition](http://www.pointclouds.org/documentation/tutorials/global_hypothesis_verification.php#global-hypothesis-verification)

..

## 11. Registration 点云注册

- 11.1 [The PCL Registration API](http://www.pointclouds.org/documentation/tutorials/registration_api.php#registration-api)

点云注册就是把一系列的点云根据重合关系拼接在一起的过程。

配对注册的一种方法：

1. 寻找点云中的兴趣点、关键点
2. 计算关键点的特征描述
3. 特征点对应关系估计
4. 丢弃部分对应特征点
5. 根据两个点云中匹配的特征点计算出变换矩阵


- 11.2 [How to use iterative closest point](http://www.pointclouds.org/documentation/tutorials/iterative_closest_point.php#iterative-closest-point)

使用两个只有 5 个点的点云演示如何使用 ICP。

- 11.3 [How to incrementally register pairs of clouds](http://www.pointclouds.org/documentation/tutorials/pairwise_incremental_registration.php#pairwise-incremental-registration)

增量式对注册。

- 11.4 [Interactive Iterative Closest Point](http://www.pointclouds.org/documentation/tutorials/interactive_icp.php#interactive-icp)

互动的方式，一步步执行 ICP 过程并可视化。

- 11.5 [How to use Normal Distributions Transform](http://www.pointclouds.org/documentation/tutorials/normal_distributions_transform.php#normal-distributions-transform)

使用 正态分布变换 Normal Distributions Transform NDT 算法，对大型点云（两种都大于 100000 个点）进行注册。

- 11.6 [In-hand scanner for small objects](http://www.pointclouds.org/documentation/tutorials/in_hand_scanner.php#in-hand-scanner)

介绍了手持扫描仪扫描小物件的程序构成。

- 11.7 [Robust pose estimation of rigid objects](http://www.pointclouds.org/documentation/tutorials/alignment_prerejective.php#alignment-prerejective)

介绍了稳健的估计物体姿态方法。

## 12. Sample Consensus 抽样一致

- 12.1 [How to use Random Sample Consensus model](http://www.pointclouds.org/documentation/tutorials/random_sample_consensus.php#random-sample-consensus)

RANSAC (RANdom SAmple Consensus) 随机抽样一致算法的使用。

在预先知道模型（如平面）的情况下，利用 RANSAC 可以有效去除干扰，得到更好的模型拟合。

## 13. Segmentation 分割

- 13.1 [Plane model segmentation](http://www.pointclouds.org/documentation/tutorials/planar_segmentation.php#planar-segmentation)

平面模型分割，区分出哪些点在平面上，哪些不在。

- 13.2 [Cylinder model segmentation](http://www.pointclouds.org/documentation/tutorials/cylinder_segmentation.php#cylinder-segmentation)

柱面模分割。

- 13.3 [Euclidean Cluster Extraction](http://www.pointclouds.org/documentation/tutorials/cluster_extraction.php#cluster-extraction)

欧几里得聚类分割，将空间上连续聚类的点进行分割，得到分离的不同的实物对应点云。

- 13.4 [Region growing segmentation](http://www.pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation)

pcl::RegionGrowing 用法。连续区域分割，基于法向。比如一个正方体的各个面就会被分割成不同的聚类。

- 13.5 [Color-based region growing segmentation](http://www.pointclouds.org/documentation/tutorials/region_growing_rgb_segmentation.php#region-growing-rgb-segmentation)

与 Region Growing Segmentation 类似，只是基于颜色信息进行分割。

- 13.6 [Min-Cut Based Segmentation](http://www.pointclouds.org/documentation/tutorials/min_cut_segmentation.php#min-cut-segmentation)

最小割分割算法，进行二值分割。基于图论。

- 13.7 [Conditional Euclidean Clustering](http://www.pointclouds.org/documentation/tutorials/conditional_euclidean_clustering.php#conditional-euclidean-clustering)

pcl::ConditionalEuclideanClustering 基于欧式距离和用户定义条件的分割算法。与 EuclideanClusterExtraction，RegionGrowingSegmentation 和 ColorBasedRegionGrowingSegmentation 相同的方法，它的优势是没有使用单一的 Euclidean，Smoothness，RGB特征进行分割，而是让用户选择综合使用这些特征。

- 13.8 [Difference of Normals Based Segmentation](http://www.pointclouds.org/documentation/tutorials/don_segmentation.php#don-segmentation)

pcl::DifferenceOfNormalsEstimation 基于 Difference of Normals features 进行分割。

1. 用一个大的支持半径 $r_l$ 对每一个点进行法向预估。
2. 用一个小的支持半径 $r_s$ 对每一个点进行法向预估。
3. 计算 DoN。
4. 根据 Scale/Region of interest 进行过滤。

- 13.9 [Clustering of Pointclouds into Supervoxels - Theoretical primer](http://www.pointclouds.org/documentation/tutorials/supervoxel_clustering.php#supervoxel-clustering)

Supervoxels..

- 13.10 [Identifying ground returns using ProgressiveMorphologicalFilter segmentation](http://www.pointclouds.org/documentation/tutorials/progressive_morphological_filtering.php#progressive-morphological-filtering)

..

- 13.11 [Filtering a PointCloud using ModelOutlierRemoval](http://www.pointclouds.org/documentation/tutorials/model_outlier_removal.php#model-outlier-removal)

在模型中使用 RANSAC 算法进行分割。

## 14. Surface

- 14.1 [Smoothing and normal estimation based on polynomial reconstruction](http://www.pointclouds.org/documentation/tutorials/resampling.php#moving-least-squares)

基于多项式重建的平滑处理和法向估计，使用的方法是 **Moving Least Squares (MLS)** 表面重建算法。进行 ReSample 后的模型更加平滑。

- 14.2 [Construct a concave or convex hull polygon for a plane model](http://www.pointclouds.org/documentation/tutorials/hull_2d.php#hull-2d)

对平面模型进行构建外壳多边形。


- 14.3 [Fast triangulation of unordered point clouds](http://www.pointclouds.org/documentation/tutorials/greedy_projection.php#greedy-triangulation)

无序点云的快速三角化。使用贪心面三角化算法，对有法向的点云，获取三角网格。

GreedyProjectionTriangulation 以及设置的一些参数。

- 14.4 [Fitting trimmed B-splines to unordered point clouds](http://www.pointclouds.org/documentation/tutorials/bspline_fitting.php#bspline-fitting)

无序点云 B 样条拟合。

## 15. Visualization 可视化

- 15.1 [The CloudViewer](http://www.pointclouds.org/documentation/tutorials/cloud_viewer.php#cloud-viewer)

```cpp
#include <pcl/visualization/cloud_viewer.h>
//...
void
foo ()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    //... populate cloud
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);
    while (!viewer.wasStopped ())
    {
    }
}
```

- 15.2 [How to visualize a range image](http://www.pointclouds.org/documentation/tutorials/range_image_visualization.php#range-image-visualization)

距离图的可视化的两种方式：点云类型 RangeImage 在 3D viewer 中显示；二维图像中使用不同颜色表示不同的距离。

- 15.3 [PCLVisualizer](http://www.pointclouds.org/documentation/tutorials/pcl_visualizer.php#pcl-visualizer)

PCLVisualizer 是比 CloudViewer 更加强大的可视化工具。

- 15.4 [PCLPlotter](http://www.pointclouds.org/documentation/tutorials/pcl_plotter.php#pcl-plotter)

PCLPlotter..

- 15.5 [Visualization](http://www.pointclouds.org/documentation/tutorials/walkthrough.php#visualization)

PLC 的 Visualization 库与 OpenCV 的 highgui 类似，为用户提供一个快速原型显示和可视化的功能：

1. 渲染和设置显示属性
2. 画基本的 3D Shape
3. 2D 直方图
4. a multitude of Geometry and Color handlers for pcl::PointCloud<T> datasets
5. a pcl::RangeImage visualization module

- 15.6 [Create a PCL visualizer in Qt with cmake](http://www.pointclouds.org/documentation/tutorials/qt_visualizer.php#qt-visualizer)

https://github.com/PointCloudLibrary/pcl/tree/master/apps 中有许多例子。

- 15.7 [Create a PCL visualizer in Qt to colorize clouds](http://www.pointclouds.org/documentation/tutorials/qt_colorize_cloud.php#qt-colorize-cloud)

..

## 16. Applications 应用

- 16.1 [Aligning object templates to a point cloud](http://www.pointclouds.org/documentation/tutorials/template_alignment.php#template-alignment)

将一个人脸的模型对齐到点云中。

- 16.2 [Cluster Recognition and 6DOF Pose Estimation using VFH descriptors](http://www.pointclouds.org/documentation/tutorials/vfh_recognition.php#vfh-recognition)

利用 VFH descriptors 对杯子的进行聚类识别和姿态分析。

- 16.3 [Point Cloud Streaming to Mobile Devices with Real-time Visualization](http://www.pointclouds.org/documentation/tutorials/mobile_streaming.php#mobile-streaming)

利用 Kinect 采集数据，然后将点云数据推流到手机上进行可视化。

- 16.4 [Detecting people on a ground plane with RGB-D data](http://www.pointclouds.org/documentation/tutorials/ground_based_rgbd_people_detection.php#ground-based-rgbd-people-detection)

RGB-D 数据人体识别。

## 17. GPU

- 17.1 [Configuring your PC to use your Nvidia GPU with PCL](http://www.pointclouds.org/documentation/tutorials/)

编译 PCL 的 GPU 支持。

- 17.2 [Using Kinfu Large Scale to generate a textured mesh](http://www.pointclouds.org/documentation/tutorials/using_kinfu_large_scale.php#using-kinfu-large-scale)

KinctFusion 的开源实现。

- 17.3 [Detecting people and their poses using PointCloud Library](http://www.pointclouds.org/documentation/tutorials/gpu_people.php#gpu-people)

人体姿态识别。

