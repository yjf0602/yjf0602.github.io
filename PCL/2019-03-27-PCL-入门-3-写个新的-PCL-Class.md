# PCL 入门-3-写个新的 PCL Class


介绍以 PCL 的风格写新的 PCL class，对应[官方tutorial](http://www.pointclouds.org/documentation/tutorials/writing_new_classes.php#writing-new-classes)。



## 1. 为什么要对 PCL 进行贡献？

回答这个问题首先假设你已经是 PCL 的使用者，PCL 作为一个工具在你的项目中起到了作用。

大多数的开源项目都是志愿行为的成果，开发者分布在全球各地，结果就是开发过程一般都有固定增量、不断迭代的特点，这就意味着：

- 开发者几乎不可能提前预料到新的代码会被用到什么情况下...
- 考虑到一些偏门的使用情况以及捕获 bug 是比较困难的，因为这些工作一般都是在业余时间弄得，缺少资源时间。

因此，所有人都会遇到需要的解决方法缺失的情况。那么解决这一问题的方法很自然地就是：**修改现有的代码去适应应用和问题**。

在讨论怎样进行代码贡献之前，还是先讨论为什么进行贡献。

在我们看来，对开源项目进行贡献有许多好处。引用 *Eric Raymond’s Linus’s Law:"given enough eyeballs, all bugs are shallow“*。这句话的意思是，当代码被公开出来，允许其他人看，那么它被修复和优化的可能性就越高。

另外，你贡献的代码可能完成了很多的事：

- 其他人基于你的代码创造了新的成果
- 你了解到了新的使用方法（你在设计的时候并没有想到它可以这样使用）
- 不用担心维护
- 你的社会名声变好 - 没有人不喜欢 free 的东西

## 2. 例子：a bilateral filter

为了更好地展示代码转换过程，我们选择这样一个例子：应用一个双边滤波器到输入点云，并保存结果到磁盘。

### 2.1 直接应用代码

```cpp
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

typedef pcl::PointXYZI PointT;

float
G (float x, float sigma)
{
  return exp (- (x*x)/(2*sigma*sigma));
}

int
main (int argc, char *argv[])
{
  std::string incloudfile = argv[1];
  std::string outcloudfile = argv[2];
  float sigma_s = atof (argv[3]);
  float sigma_r = atof (argv[4]);

  // Load cloud
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::io::loadPCDFile (incloudfile.c_str (), *cloud);
  int pnumber = (int)cloud->size ();

  // Output Cloud = Input Cloud
  pcl::PointCloud<PointT> outcloud = *cloud;

  // Set up KDTree
  pcl::KdTreeFLANN<PointT>::Ptr tree (new pcl::KdTreeFLANN<PointT>);
  tree->setInputCloud (cloud);

  // Neighbors containers
  std::vector<int> k_indices;
  std::vector<float> k_distances;

  // Main Loop
  for (int point_id = 0; point_id < pnumber; ++point_id)
  {
    float BF = 0;
    float W = 0;

    tree->radiusSearch (point_id, 2 * sigma_s, k_indices, k_distances);

    // For each neighbor
    for (size_t n_id = 0; n_id < k_indices.size (); ++n_id)
    {
      float id = k_indices.at (n_id);
      float dist = sqrt (k_distances.at (n_id));
      float intensity_dist = abs (cloud->points[point_id].intensity - cloud->points[id].intensity);

      float w_a = G (dist, sigma_s);
      float w_b = G (intensity_dist, sigma_r);
      float weight = w_a * w_b;

      BF += weight * cloud->points[id].intensity;
      W += weight;
    }

    outcloud.points[point_id].intensity = BF / W;
  }

  // Save filtered output
  pcl::io::savePCDFile (outcloudfile.c_str (), outcloud);
  return (0);
}
```

### 2.2 设置文件结构

有两种方式设置文件结构：1) 使代码分开，作为一个独立的 PCL class，不在 PCL 的代码树中。2) 将代码添加到 PCL 的代码树中。由于我们假设代码会被贡献到 PCL 中，因此注意力集中到后面这种方式，这种方式也是更加复杂一些。

假设我们希望把新的算法添加到 PCL filter 的代码目录，首先需要添加3个文件到 filters 目录中：

- include/pcl/filters/bilateral.h - 包含所有的定义
- include/pcl/filters/impl/bilateral.hpp - 包含模板的实现
- src/bilateral.cpp - 包含模板显式实例化代码

同时，我们需要给我们的算法起一个名字，就叫它 *BilateralFilter*。

>有一些 PLC 滤波算法提供了两种实现，一种用 PointCloud<T> 类型，另一种用遗留的 PLCPointCloud2 类型。这个不再要求了。

#### **bilateral.h**

*bilateral.h* 包含 *BilateralFilter class* 的相关定义。这是一个最小的结构：

```cpp
#ifndef PCL_FILTERS_BILATERAL_H_
#define PCL_FILTERS_BILATERAL_H_

#include <pcl/filters/filter.h>

namespace pcl
{
  template<typename PointT>
  class BilateralFilter : public Filter<PointT>
  {
  };
}

#endif // PCL_FILTERS_BILATERAL_H_
```

#### **bilateral.hpp**

```cpp
#ifndef PCL_FILTERS_BILATERAL_IMPL_H_
#define PCL_FILTERS_BILATERAL_IMPL_H_

#include <pcl/filters/bilateral.h>

#endif // PCL_FILTERS_BILATERAL_IMPL_H_
```

暂时还没有添加具体的实现。

#### **bilateral.cpp**

```cpp
#include <pcl/filters/bilateral.h>
#include <pcl/filters/impl/bilateral.hpp>
```

#### **CMakeLists.txt**

```CMakeLists.txt
# Find "set (srcs", and add a new entry there, e.g.,
 set (srcs
      src/conditional_removal.cpp
      # ...
      src/bilateral.cpp)
      )

 # Find "set (incs", and add a new entry there, e.g.,
 set (incs
      include pcl/${SUBSYS_NAME}/conditional_removal.h
      # ...
      include pcl/${SUBSYS_NAME}/bilateral.h
      )

 # Find "set (impl_incs", and add a new entry there, e.g.,
 set (impl_incs
      include/pcl/${SUBSYS_NAME}/impl/conditional_removal.hpp
      # ...
      include/pcl/${SUBSYS_NAME}/impl/bilateral.hpp
      )
```

### 2.3 往文件里添加内容

如果已经按上面的结构添加文件，重新编译 PCL 应该就没有问题了。首先从 *bilateral.cpp* 开始，它的内容是最少的。

#### **bilateral.cpp**

如前所提到的，我们要对需要使用的 PointT 进行显示实例化。最简单的方式就为每个需要支持的 PointT 进行显式实例化声明：

```cpp
#include <pcl/point_types.h>
 #include <pcl/filters/bilateral.h>
 #include <pcl/filters/impl/bilateral.hpp>

 template class PCL_EXPORTS pcl::BilateralFilter<pcl::PointXYZ>;
 template class PCL_EXPORTS pcl::BilateralFilter<pcl::PointXYZI>;
 template class PCL_EXPORTS pcl::BilateralFilter<pcl::PointXYZRGB>;
 // ...
```

另外也可以使用 *PCL_INSTANTIATE* 宏进行：

```cpp
#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/impl/bilateral.hpp>

PCL_INSTANTIATE(BilateralFilter, PCL_XYZ_POINT_TYPES);
```

上面就是对所有包含 XYZ 数据的点类型进行显式实例化了。但是在我们现在的例子里面，使用了 intensity 数据，因此，只能对包含 xyz 数据和 intensity 数据的点类型进行显式实例化。最终的 *bilateral.cpp* 应该是这样：

```cpp
#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/impl/bilateral.hpp>

PCL_INSTANTIATE(BilateralFilter, (pcl::PointXYZI)(pcl::PointXYZINormal));
```

#### **bilateral.h**

在这个文件中添加构造函数以及成员变量，因为有两个成员变量，同时为他们加上 *setters* 和 *getters*。

```cpp
...
namespace pcl
{
  template<typename PointT>
  class BilateralFilter : public Filter<PointT>
  {
    public:
      BilateralFilter () : sigma_s_ (0),
                          sigma_r_ (std::numeric_limits<double>::max ())
      {
      }

      void
      setSigmaS (const double sigma_s)
      {
        sigma_s_ = sigma_s;
      }

      double
      getSigmaS () const
      {
        return (sigma_s_);
      }

      void
      setSigmaR (const double sigma_r)
      {
        sigma_r_ = sigma_r;
      }

      double
      getSigmaR () const
      {
        return (sigma_r_);
      }

    private:
      double sigma_s_;
      double sigma_r_;
  };
}

#endif // PCL_FILTERS_BILATERAL_H_
```

在构造函数中，给两个成员变量进行了初始化。因为我们的类是继承 pcl::Filter 的，它又是继承 pcl::PCLBase 的，所以使用 pcl::PCLBase::SetInputCloud 方法把输入数据传递给我们的算法。因此我们添加一个 using 声明：

```cpp
...
  template<typename PointT>
  class BilateralFilter : public Filter<PointT>
  {
    using Filter<PointT>::input_;
    public:
      BilateralFilter () : sigma_s_ (0),
...
```

这样我们的类就可以直接使用 input_ 而不用进行完整的声明。接着，继承了 pcl::Filter 的类必须实现 pcl::Filter::applyFilter 方法，因此定义：

```cpp
...
    using Filter<PointT>::input_;
    typedef typename Filter<PointT>::PointCloud PointCloud;

    public:
      BilateralFilter () : sigma_s_ (0),
                          sigma_r_ (std::numeric_limits<double>::max ())
      {
      }

      void
      applyFilter (PointCloud &output);
...
```

applyFilter 方法将在 bilateral.hpp 文件中实现。上面的第三行，构造了一个类型定义，这样就可以直接使用 PointCloud 而不用使用完整的定义。

在前面的 bilateral 应用中，对与点云中的每个点都进行了同样的操作，为了使 applyFilter 变得简洁，我们定义一个方法 computPointWeight 来实现这些相同的操作：

```cpp
...
      void
      applyFilter (PointCloud &output);

      double
      computePointWeight (const int pid, const std::vector<int> &indices, const std::vector<float> &distances);
...
```

同样的，在 bilateral 应用中，用到了 pcl::KdTree 以及相同的操作，我们添加：

```cpp
#include <pcl/kdtree/kdtree.h>
...
    using Filter<PointT>::input_;
    typedef typename Filter<PointT>::PointCloud PointCloud;
    typedef typename pcl::KdTree<PointT>::Ptr KdTreePtr;

  public:
...

    void
    setSearchMethod (const KdTreePtr &tree)
    {
      tree_ = tree;
    }

  private:
...
    KdTreePtr tree_;
...
```

最后，再把 G(flaot x, float sigma) 内联。因为这个函数只在内部使用，声明它为 private。头文件最终变成：

```cpp
#ifndef PCL_FILTERS_BILATERAL_H_
#define PCL_FILTERS_BILATERAL_H_

#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree.h>

namespace pcl
{
  template<typename PointT>
  class BilateralFilter : public Filter<PointT>
  {
    using Filter<PointT>::input_;
    typedef typename Filter<PointT>::PointCloud PointCloud;
    typedef typename pcl::KdTree<PointT>::Ptr KdTreePtr;

    public:
      BilateralFilter () : sigma_s_ (0),
                          sigma_r_ (std::numeric_limits<double>::max ())
      {
      }


      void
      applyFilter (PointCloud &output);

      double
      computePointWeight (const int pid, const std::vector<int> &indices, const std::vector<float> &distances);

      void
      setSigmaS (const double sigma_s)
      {
        sigma_s_ = sigma_s;
      }

      double
      getSigmaS () const
      {
        return (sigma_s_);
      }

      void
      setSigmaR (const double sigma_r)
      {
        sigma_r_ = sigma_r;
      }

      double
      getSigmaR () const
      {
        return (sigma_r_);
      }

      void
      setSearchMethod (const KdTreePtr &tree)
      {
        tree_ = tree;
      }


    private:

      inline double
      kernel (double x, double sigma)
      {
        return (exp (- (x*x)/(2*sigma*sigma)));
      }

      double sigma_s_;
      double sigma_r_;
      KdTreePtr tree_;
  };
}

#endif // PCL_FILTERS_BILATERAL_H_
```

#### **bilateral.hpp**

这里需要实现两个方法，*applyFilter* 和 *computePointWeight*。

```cpp
template <typename PointT> double
pcl::BilateralFilter<PointT>::computePointWeight (const int pid,
                                                  const std::vector<int> &indices,
                                                  const std::vector<float> &distances)
{
  double BF = 0, W = 0;

  // For each neighbor
  for (size_t n_id = 0; n_id < indices.size (); ++n_id)
  {
    double id = indices[n_id];
    double dist = std::sqrt (distances[n_id]);
    double intensity_dist = abs (input_->points[pid].intensity - input_->points[id].intensity);

    double weight = kernel (dist, sigma_s_) * kernel (intensity_dist, sigma_r_);

    BF += weight * input_->points[id].intensity;
    W += weight;
  }
  return (BF / W);
}

template <typename PointT> void
pcl::BilateralFilter<PointT>::applyFilter (PointCloud &output)
{
  tree_->setInputCloud (input_);

  std::vector<int> k_indices;
  std::vector<float> k_distances;

  output = *input_;

  for (size_t point_id = 0; point_id < input_->points.size (); ++point_id)
  {
    tree_->radiusSearch (point_id, sigma_s_ * 2, k_indices, k_distances);

    output.points[point_id].intensity = computePointWeight (point_id, k_indices, k_distances);
  }

}
```

现在是时间声明 PCL_INSTANTIATE entry:

```cpp
#ifndef PCL_FILTERS_BILATERAL_IMPL_H_
#define PCL_FILTERS_BILATERAL_IMPL_H_

#include <pcl/filters/bilateral.h>

...

#define PCL_INSTANTIATE_BilateralFilter(T) template class PCL_EXPORTS pcl::BilateralFilter<T>;

#endif // PCL_FILTERS_BILATERAL_IMPL_H_
```

还有一件可以做的事是进行错误检查：

- sigma_a 和 sigma_r 参数是否给了
- 搜索方法是否设置

使用 pcl::PCL_ERROR 宏定义进行检查。还有就是可以根据点云是否有序选择更加有效的搜索方法：

```cpp
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/organized_data.h>

...
template <typename PointT> void
pcl::BilateralFilter<PointT>::applyFilter (PointCloud &output)
{
  if (sigma_s_ == 0)
  {
    PCL_ERROR ("[pcl::BilateralFilter::applyFilter] Need a sigma_s value given before continuing.\n");
    return;
  }
  if (!tree_)
  {
    if (input_->isOrganized ())
      tree_.reset (new pcl::OrganizedNeighbor<PointT> ());
    else
      tree_.reset (new pcl::KdTreeFLANN<PointT> (false));
  }
  tree_->setInputCloud (input_);
...
```

完整的实现头文件如下：

```cpp
#ifndef PCL_FILTERS_BILATERAL_IMPL_H_
#define PCL_FILTERS_BILATERAL_IMPL_H_

#include <pcl/filters/bilateral.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/organized_data.h>

template <typename PointT> double
pcl::BilateralFilter<PointT>::computePointWeight (const int pid,
                                                  const std::vector<int> &indices,
                                                  const std::vector<float> &distances)
{
  double BF = 0, W = 0;

  // For each neighbor
  for (size_t n_id = 0; n_id < indices.size (); ++n_id)
  {
    double id = indices[n_id];
    double dist = std::sqrt (distances[n_id]);
    double intensity_dist = abs (input_->points[pid].intensity - input_->points[id].intensity);

    double weight = kernel (dist, sigma_s_) * kernel (intensity_dist, sigma_r_);

    BF += weight * input_->points[id].intensity;
    W += weight;
  }
  return (BF / W);
}

template <typename PointT> void
pcl::BilateralFilter<PointT>::applyFilter (PointCloud &output)
{
  if (sigma_s_ == 0)
  {
    PCL_ERROR ("[pcl::BilateralFilter::applyFilter] Need a sigma_s value given before continuing.\n");
    return;
  }
  if (!tree_)
  {
    if (input_->isOrganized ())
      tree_.reset (new pcl::OrganizedNeighbor<PointT> ());
    else
      tree_.reset (new pcl::KdTreeFLANN<PointT> (false));
  }
  tree_->setInputCloud (input_);

  std::vector<int> k_indices;
  std::vector<float> k_distances;

  output = *input_;

  for (size_t point_id = 0; point_id < input_->points.size (); ++point_id)
  {
    tree_->radiusSearch (point_id, sigma_s_ * 2, k_indices, k_distances);

    output.points[point_id].intensity = computePointWeight (point_id, k_indices, k_distances);
  }
}

#define PCL_INSTANTIATE_BilateralFilter(T) template class PCL_EXPORTS pcl::BilateralFilter<T>;

#endif // PCL_FILTERS_BILATERAL_IMPL_H_
```

## 3. 有效利用 PCL 的一些概念

### Point indices

标准的传递点云数据给 PCL 算法的方式是调用 pcl::SetInputCloud。另外，PCL 也定义了一个感兴趣区域 ROI（Region of interest）/ List of
Point indices 来表明算法是否要对该点进行操作。这可以通过 pcl::PCLBase::setIndices 设置。

因此，新的 bilateral.hpp:

```cpp
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/organized_data.h>

...
template <typename PointT> void
pcl::BilateralFilter<PointT>::applyFilter (PointCloud &output)
{
  if (sigma_s_ == 0)
  {
    PCL_ERROR ("[pcl::BilateralFilter::applyFilter] Need a sigma_s value given before continuing.\n");
    return;
  }
  if (!tree_)
  {
    if (input_->isOrganized ())
      tree_.reset (new pcl::OrganizedNeighbor<PointT> ());
    else
      tree_.reset (new pcl::KdTreeFLANN<PointT> (false));
  }
  tree_->setInputCloud (input_);
...
```

新的实现文件：

```cpp
#ifndef PCL_FILTERS_BILATERAL_IMPL_H_
#define PCL_FILTERS_BILATERAL_IMPL_H_

#include <pcl/filters/bilateral.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/organized_data.h>

template <typename PointT> double
pcl::BilateralFilter<PointT>::computePointWeight (const int pid,
                                                  const std::vector<int> &indices,
                                                  const std::vector<float> &distances)
{
  double BF = 0, W = 0;

  // For each neighbor
  for (size_t n_id = 0; n_id < indices.size (); ++n_id)
  {
    double id = indices[n_id];
    double dist = std::sqrt (distances[n_id]);
    double intensity_dist = abs (input_->points[pid].intensity - input_->points[id].intensity);

    double weight = kernel (dist, sigma_s_) * kernel (intensity_dist, sigma_r_);

    BF += weight * input_->points[id].intensity;
    W += weight;
  }
  return (BF / W);
}

template <typename PointT> void
pcl::BilateralFilter<PointT>::applyFilter (PointCloud &output)
{
  if (sigma_s_ == 0)
  {
    PCL_ERROR ("[pcl::BilateralFilter::applyFilter] Need a sigma_s value given before continuing.\n");
    return;
  }
  if (!tree_)
  {
    if (input_->isOrganized ())
      tree_.reset (new pcl::OrganizedNeighbor<PointT> ());
    else
      tree_.reset (new pcl::KdTreeFLANN<PointT> (false));
  }
  tree_->setInputCloud (input_);

  std::vector<int> k_indices;
  std::vector<float> k_distances;

  output = *input_;

  for (size_t i = 0; i < indices_->size (); ++i)
  {
    tree_->radiusSearch ((*indices_)[i], sigma_s_ * 2, k_indices, k_distances);

    output.points[(*indices_)[i]].intensity = computePointWeight ((*indices_)[i], k_indices, k_distances);
  }
}

#define PCL_INSTANTIATE_BilateralFilter(T) template class PCL_EXPORTS pcl::BilateralFilter<T>;

#endif // PCL_FILTERS_BILATERAL_IMPL_H_
```

同样的，为了方便，声明 indices_：

```cpp
...
  template<typename PointT>
  class BilateralFilter : public Filter<PointT>
  {
    using Filter<PointT>::input_;
    using Filter<PointT>::indices_;
    public:
      BilateralFilter () : sigma_s_ (0),
...
```

### 协议

BSD License。

### 合理命名

setter getter 命名应该更加清晰。

set/getSigmaS 改为 set/getHalfSize。 set/getSigmaR 改为 set/getStdDev。

### 代码注释

PCL 使用 Doxygen 风格。

bilateral.h：

```cpp
/*
* Software License Agreement (BSD License)
*
*  Point Cloud Library (PCL) - www.pointclouds.org
*  Copyright (c) 2010-2011, Willow Garage, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/

#ifndef PCL_FILTERS_BILATERAL_H_
#define PCL_FILTERS_BILATERAL_H_

#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree.h>

namespace pcl
{
  /** \brief A bilateral filter implementation for point cloud data. Uses the intensity data channel.
    * \note For more information please see
    * <b>C. Tomasi and R. Manduchi. Bilateral Filtering for Gray and Color Images.
    * In Proceedings of the IEEE International Conference on Computer Vision,
    * 1998.</b>
    * \author Luca Penasa
    */
  template<typename PointT>
  class BilateralFilter : public Filter<PointT>
  {
    using Filter<PointT>::input_;
    using Filter<PointT>::indices_;
    typedef typename Filter<PointT>::PointCloud PointCloud;
    typedef typename pcl::KdTree<PointT>::Ptr KdTreePtr;

    public:
      /** \brief Constructor.
        * Sets \ref sigma_s_ to 0 and \ref sigma_r_ to MAXDBL
        */
      BilateralFilter () : sigma_s_ (0),
                          sigma_r_ (std::numeric_limits<double>::max ())
      {
      }


      /** \brief Filter the input data and store the results into output
        * \param[out] output the resultant point cloud message
        */
      void
      applyFilter (PointCloud &output);

      /** \brief Compute the intensity average for a single point
        * \param[in] pid the point index to compute the weight for
        * \param[in] indices the set of nearest neighor indices
        * \param[in] distances the set of nearest neighbor distances
        * \return the intensity average at a given point index
        */
      double
      computePointWeight (const int pid, const std::vector<int> &indices, const std::vector<float> &distances);

      /** \brief Set the half size of the Gaussian bilateral filter window.
        * \param[in] sigma_s the half size of the Gaussian bilateral filter window to use
        */
      inline void
      setHalfSize (const double sigma_s)
      {
        sigma_s_ = sigma_s;
      }

      /** \brief Get the half size of the Gaussian bilateral filter window as set by the user. */
      double
      getHalfSize () const
      {
        return (sigma_s_);
      }

      /** \brief Set the standard deviation parameter
        * \param[in] sigma_r the new standard deviation parameter
        */
      void
      setStdDev (const double sigma_r)
      {
        sigma_r_ = sigma_r;
      }

      /** \brief Get the value of the current standard deviation parameter of the bilateral filter. */
      double
      getStdDev () const
      {
        return (sigma_r_);
      }

      /** \brief Provide a pointer to the search object.
        * \param[in] tree a pointer to the spatial search object.
        */
      void
      setSearchMethod (const KdTreePtr &tree)
      {
        tree_ = tree;
      }

    private:

      /** \brief The bilateral filter Gaussian distance kernel.
        * \param[in] x the spatial distance (distance or intensity)
        * \param[in] sigma standard deviation
        */
      inline double
      kernel (double x, double sigma)
      {
        return (exp (- (x*x)/(2*sigma*sigma)));
      }

      /** \brief The half size of the Gaussian bilateral filter window (e.g., spatial extents in Euclidean). */
      double sigma_s_;
      /** \brief The standard deviation of the bilateral filter (e.g., standard deviation in intensity). */
      double sigma_r_;

      /** \brief A pointer to the spatial search object. */
      KdTreePtr tree_;
  };
}

#endif // PCL_FILTERS_BILATERAL_H_
```

bilateral.hpp：

```cpp
/*
* Software License Agreement (BSD License)
*
*  Point Cloud Library (PCL) - www.pointclouds.org
*  Copyright (c) 2010-2011, Willow Garage, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/

#ifndef PCL_FILTERS_BILATERAL_IMPL_H_
#define PCL_FILTERS_BILATERAL_IMPL_H_

#include <pcl/filters/bilateral.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/organized_data.h>

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> double
pcl::BilateralFilter<PointT>::computePointWeight (const int pid,
                                                  const std::vector<int> &indices,
                                                  const std::vector<float> &distances)
{
  double BF = 0, W = 0;

  // For each neighbor
  for (size_t n_id = 0; n_id < indices.size (); ++n_id)
  {
    double id = indices[n_id];
    // Compute the difference in intensity
    double intensity_dist = abs (input_->points[pid].intensity - input_->points[id].intensity);

    // Compute the Gaussian intensity weights both in Euclidean and in intensity space
    double dist = std::sqrt (distances[n_id]);
    double weight = kernel (dist, sigma_s_) * kernel (intensity_dist, sigma_r_);

    // Calculate the bilateral filter response
    BF += weight * input_->points[id].intensity;
    W += weight;
  }
  return (BF / W);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::BilateralFilter<PointT>::applyFilter (PointCloud &output)
{
  // Check if sigma_s has been given by the user
  if (sigma_s_ == 0)
  {
    PCL_ERROR ("[pcl::BilateralFilter::applyFilter] Need a sigma_s value given before continuing.\n");
    return;
  }
  // In case a search method has not been given, initialize it using some defaults
  if (!tree_)
  {
    // For organized datasets, use an OrganizedNeighbor
    if (input_->isOrganized ())
      tree_.reset (new pcl::OrganizedNeighbor<PointT> ());
    // For unorganized data, use a FLANN kdtree
    else
      tree_.reset (new pcl::KdTreeFLANN<PointT> (false));
  }
  tree_->setInputCloud (input_);

  std::vector<int> k_indices;
  std::vector<float> k_distances;

  // Copy the input data into the output
  output = *input_;

  // For all the indices given (equal to the entire cloud if none given)
  for (size_t i = 0; i < indices_->size (); ++i)
  {
    // Perform a radius search to find the nearest neighbors
    tree_->radiusSearch ((*indices_)[i], sigma_s_ * 2, k_indices, k_distances);

    // Overwrite the intensity value with the computed average
    output.points[(*indices_)[i]].intensity = computePointWeight ((*indices_)[i], k_indices, k_distances);
  }
}

#define PCL_INSTANTIATE_BilateralFilter(T) template class PCL_EXPORTS pcl::BilateralFilter<T>;

#endif // PCL_FILTERS_BILATERAL_IMPL_H_
```


### 测试新的 class

```cpp
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/bilateral.h>

typedef pcl::PointXYZI PointT;

int
main (int argc, char *argv[])
{
  std::string incloudfile = argv[1];
  std::string outcloudfile = argv[2];
  float sigma_s = atof (argv[3]);
  float sigma_r = atof (argv[4]);

  // Load cloud
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::io::loadPCDFile (incloudfile.c_str (), *cloud);

  pcl::PointCloud<PointT> outcloud;

  // Set up KDTree
  pcl::KdTreeFLANN<PointT>::Ptr tree (new pcl::KdTreeFLANN<PointT>);

  pcl::BilateralFilter<PointT> bf;
  bf.setInputCloud (cloud);
  bf.setSearchMethod (tree);
  bf.setHalfSize (sigma_s);
  bf.setStdDev (sigma_r);
  bf.filter (outcloud);

  // Save filtered output
  pcl::io::savePCDFile (outcloudfile.c_str (), outcloud);
  return (0);
}
```



