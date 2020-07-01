# SGM 论文理解


Opencv 中的立体匹配算法 SGBM 是根据论文《Stereo Processing by Semiglobal Matching and Mutual Information》 实现的，本文主要就是记录对该论文的理解。


## 1. 介绍

第一节首先主要介绍了目前 Stereo 领域的一些研究成果，介绍了算法分类及优缺点。

**算法分类论文**：

- www.middlebury.edu/stereo
- [1] D. Scharstein and R. Szeliski, “A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms,” Int’l J. Computer Vision, vol. 47, nos. 1/2/3, pp. 7-42, Apr.-June 2002.

**代价计算**

- [2] S. Birchfield and C. Tomasi, “Depth Discontinuities by Pixel-to-Pixel Stereo,” Proc. Sixth IEEE Int’l Conf. Computer Vision, pp. 1073-1080, Jan. 1998. **AD/SD...**
- [3] A. Klaus, M. Sormann, and K. Karner, “Segment-Based Stereo Matching Using Belief Propagation and a Self-Adapting Dissimilarity Measure,” Proc. Int’l Conf. Pattern Recognition, 2006. **Mutual information**
  
**固定窗口代价聚合 costs are simplysummedover a fixed sized window at constant disparity**

- [3] A. Klaus, M. Sormann, and K. Karner, “Segment-Based Stereo Matching Using Belief Propagation and a Self-Adapting Dissimilarity Measure,” Proc. Int’l Conf. Pattern Recognition, 2006.
- [5] G. Egnal, “Mutual Information as a Stereo Correspondence Measure,” Technical Report MS-CIS-00-20, Computer and Information Science, Univ. of Pennsylvania, 2000.
- [8] H. Hirschmu¨ ller, P.R. Innocent, and J.M. Garibaldi, “Real-Time Correlation-Based Stereo Vision with Reduced Border Errors,” Int’l J. Computer Vision, vol. 47, nos. 1/2/3, pp. 229-246, Apr.-June 2002.
- [9] M. Bleyer and M. Gelautz, “A Layered Stereo Matching Algorithm Using Image Segmentation and Global Visibility Constraints,” ISPRS J. Photogrammetry and Remote Sensing, vol. 59, no. 3, pp. 128-150, 2005.
 
 **动态调整窗口**

 - [10] K.-J. Yoon and I.-S. Kweon, “Adaptive Support-Weight Approach for Correspondence Search,” IEEE Trans. Pattern Matching and Machine Intelligence, vol. 28, no. 4, pp. 650-656, Apr. 2006.
 - [11] Q. Yang, L. Wang, R. Yang, H. Stewenius, and D. Nister, “Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation and Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
 - [7] C.L. Zitnick, S.B. Kang, M. Uyttendaele, S.Winder, and R. Szeliski, “High-Quality Video View Interpolation Using a Layered Representation,” Proc. ACM SIGGRAPH ’04, 2004.
 - [12] C. Lei, J. Selzer, and Y.-H. Yang, “Region-Tree Based Stereo Using Dynamic Programming Optimization,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.

**局部算法WTA winner-takes-all**

- [5] G. Egnal, “Mutual Information as a Stereo Correspondence Measure,” Technical Report MS-CIS-00-20, Computer and Information Science, Univ. of Pennsylvania, 2000.
- [8] H. Hirschmu¨ ller, P.R. Innocent, and J.M. Garibaldi, “Real-Time Correlation-Based Stereo Vision with Reduced Border Errors,” Int’l J. Computer Vision, vol. 47, nos. 1/2/3, pp. 229-246, Apr.-June 2002.
- [10] K.-J. Yoon and I.-S. Kweon, “Adaptive Support-Weight Approach for Correspondence Search,” IEEE Trans. Pattern Matching and Machine Intelligence, vol. 28, no. 4, pp. 650-656, Apr. 2006.

**全局算法-penalizing occllusions**

- [9] M. Bleyer and M. Gelautz, “A Layered Stereo Matching Algorithm Using Image Segmentation and Global Visibility Constraints,” ISPRS J. Photogrammetry and Remote Sensing, vol. 59, no. 3, pp. 128-150, 2005.
- [13] V. Kolmogorov and R. Zabih, “Computing Visual Correspondence with Occlusions Using Graph Cuts,” Proc. Int’l Conf. Computer Vision, vol. 2, pp. 508-515, 2001.

**全局算法-alternatively treating visibility**

- [11] Q. Yang, L. Wang, R. Yang, H. Stewenius, and D. Nister, “Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation and Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [12] C. Lei, J. Selzer, and Y.-H. Yang, “Region-Tree Based Stereo Using Dynamic Programming Optimization,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [14] J. Sun, Y. Li, S. Kang, and H.-Y. Shum, “Symmetric Stereo Matching for Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, vol. 2, pp. 399-406, June 2005.

**全局算法-enforcing a left/right or symmetric consistency between images**

- [7] C.L. Zitnick, S.B. Kang, M. Uyttendaele, S.Winder, and R. Szeliski, “High-Quality Video View Interpolation Using a Layered Representation,” Proc. ACM SIGGRAPH ’04, 2004.
- [11] Q. Yang, L. Wang, R. Yang, H. Stewenius, and D. Nister, “Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation and Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [12] C. Lei, J. Selzer, and Y.-H. Yang, “Region-Tree Based Stereo Using Dynamic Programming Optimization,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [14] J. Sun, Y. Li, S. Kang, and H.-Y. Shum, “Symmetric Stereo Matching for Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, vol. 2, pp. 399-406, June 2005.

**全局算法-weight the smoothness term according to segmentation information**

- [14] J. Sun, Y. Li, S. Kang, and H.-Y. Shum, “Symmetric Stereo Matching for Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, vol. 2, pp. 399-406, June 2005.

**全局算法-DP dynamic programming**

- [2] S. Birchfield and C. Tomasi, “Depth Discontinuities by Pixel-to-Pixel Stereo,” Proc. Sixth IEEE Int’l Conf. Computer Vision, pp. 1073-1080, Jan. 1998.
- [15] G. Van Meerbergen, M. Vergauwen, M. Pollefeys, and L. Van Gool, “A Hierarchical Symmetric Stereo Algorithm Using Dynamic Programming,” Int’l J. Computer Vision, vol. 47, nos. 1/2/3, pp. 275-285, Apr.-June 2002.

DP 算法会导致条纹效应，因此提出了下面的 Tree-based DP.

**全局算法-Tree-based DP**

- [12] C. Lei, J. Selzer, and Y.-H. Yang, “Region-Tree Based Stereo Using Dynamic Programming Optimization,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [16] O. Veksler, “Stereo Correspondence by Dynamic Programming on a Tree,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, vol. 2, pp. 384-390, June 2005.

**全局算法-Graph Cuts**

- [13] V. Kolmogorov and R. Zabih, “Computing Visual Correspondence with Occlusions Using Graph Cuts,” Proc. Int’l Conf. Computer Vision, vol. 2, pp. 508-515, 2001.

**全局算法-Belief Propagation**

- [3] A. Klaus, M. Sormann, and K. Karner, “Segment-Based Stereo Matching Using Belief Propagation and a Self-Adapting Dissimilarity Measure,” Proc. Int’l Conf. Pattern Recognition, 2006.
- [11] Q. Yang, L. Wang, R. Yang, H. Stewenius, and D. Nister, “Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation and Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [14] J. Sun, Y. Li, S. Kang, and H.-Y. Shum, “Symmetric Stereo Matching for Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, vol. 2, pp. 399-406, June 2005.

**全局算法-Layered approaches perform image segmentation and model planes in disparity space, which are interatively optimized**

- [3] A. Klaus, M. Sormann, and K. Karner, “Segment-Based Stereo Matching Using Belief Propagation and a Self-Adapting Dissimilarity Measure,” Proc. Int’l Conf. Pattern Recognition, 2006.
- [9] M. Bleyer and M. Gelautz, “A Layered Stereo Matching Algorithm Using Image Segmentation and Global Visibility Constraints,” ISPRS J. Photogrammetry and Remote Sensing, vol. 59, no. 3, pp. 128-150, 2005.
- [11] Q. Yang, L. Wang, R. Yang, H. Stewenius, and D. Nister, “Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation and Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.

**视差图优化 Disparity refinement**

- [1] D. Scharstein and R. Szeliski, “A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms,” Int’l J. Computer Vision, vol. 47, nos. 1/2/3, pp. 7-42, Apr.-June 2002.
- [8] H. Hirschmu¨ ller, P.R. Innocent, and J.M. Garibaldi, “Real-Time Correlation-Based Stereo Vision with Reduced Border Errors,” Int’l J. Computer Vision, vol. 47, nos. 1/2/3, pp. 229-246, Apr.-June 2002.
- [11] Q. Yang, L. Wang, R. Yang, H. Stewenius, and D. Nister, “Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation and Occlusion Handling,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [12] C. Lei, J. Selzer, and Y.-H. Yang, “Region-Tree Based Stereo Using Dynamic Programming Optimization,” Proc. IEEE Conf. Computer Vision and Pattern Recognition, June 2006.
- [17] H. Hirschmu¨ ller, “Stereo Vision Based Mapping and Immediate Virtual Walkthroughs,” PhD dissertation, School of Computing, De Montfort Univ., June 2003.

然后提到了**大部分匹配效果好的算法都使用了全局能量优化**，然而全局算法运算复杂，运算速度慢，接着提出了本论文的 Semiglobal Matching (SGM) method:

- 2.1 基于互信息（Mutual information）的 Cost 计算;
- 2.2 Cost 聚合; 
- 2.3 Disparity refine
- 2.4 Multibaseline matching
- 2.5 进一步的视差优化，包括尖峰滤除（peaks filter）、间隙插值（gap interpolation）
- 2.6 large image
- 2.7 使用正交投影融合多个 disparity
- 3 Result

## 2 Semiglobal Matching 半全局匹配

### 2.1 Pixelwise Matching Cost Calculation 像素级匹配代价计算

首先确保输入图像满足对极几何关系，经过了校准。


base iamge $I_{b\mathbf{p}}$ 中的 base image pixel: $\mathbf{p}$


$\mathbf{p}$ 对应的 match cost：$\mathbf{q}$

$\mathbf{q}$ 组成 $I_{m\mathbf{q}}$

其中:
$$\mathbf{q}=e_{bm}(\mathbf{p},d)$$

$$e_{bm}(\mathbf{p},d)=[p_x-d,p_y]^T$$

其中 $d$ 就是 disparity，视差。

一种 pixelwise cost 计算方法的选择是 $C_{BT}(\mathbf{p},d)$，这里介绍的是基于互信息的方法，这种方法对于光照变化不敏感。

互信息 Mutual information 是从熵 $H$ 引出的：
$$
MI_{I_1,I_2}=H_{I_1}+H_{I_2}-H_{I_1,I_2}-----(1)
$$

熵从图像的概率分布 P 进行计算：
$$
H_I=-\int_0^1P_I(i)logP_I(i)di-----（2） \\
H_{I_1,I_2}=-\int_0^1\int_0^1P_{I_1,I_2}(i_1,i_2)logP_{I_1,I_2}(i_1,i_2)di_1di_2-----(3)
$$

对于两个比较相似的图像，$H_{I_1,I_2}$ 比较小，因此互信息比较大。在立体匹配中，一个图像需要根据视差图 $D$进行变换后再用于互信息计算：$I_1=I_b, I_2=f_D(I_m)$。

上面公式（1）的计算需要一个先验的视差图，这限制了 MI 在 pixel matching cost 上的使用。

Kim et al. 将 $H_{I_1,I_2}$ 的计算通过泰勒展开转换为在像素上进行求和：

$$
H_{I_1,I_2}=\sum_\mathbf{p}h_{I_1,I_2}(I_{1\mathbf{p}},I_{2\mathbf{p}})-----(4)
$$

其中，$h_{I_1,I_2}$ 由图像的联合概率分布 $P_{I_1,I_2}$ 计算得到。n 为关联像素个数，二维高斯卷积$\otimes g(i,k)$后的结果：

$$
h_{I_1,I_2}(i,k)=-\frac{1}{n}log(P_{I_1,I_2}(i,k)\otimes g(i,k))\otimes g(i,k)-----(5)
$$

联合概率分布 $P_{I_1,I_2}(i,k)$ 的计算公式如下($T[]$判据为真时等于1，假时为0)：

$$
P_{I_1,I_2}(i,k)=\frac{1}{n}\sum_\mathbf{p}T[(i,k)=(I_{1\mathbf{p}},I_{2\mathbf{p}})]-----(6)
$$

**公式(6)为遍历所有的像素对，求出像素值为(i,k)时的概率。**

上述过程的计算可视化：

![](https://cdn.jsdelivr.net/gh/yjf0602/PicHost/img/sgm_hii.png)

类似的有：
$$
H_I=\sum_\mathbf{p}h_I(I_\mathbf{p})-----(7a) \\
h_I(i)=-\frac{1}{n}log(P_I(i)\otimes g(i))\otimes g(i)-----(7b)
$$

概率分布 $P_I$ 必须只在匹配像素对上进行，不能对整个 $I$，不然遮挡就会被忽略，$H_{I_1}$ 和 $H_{I_2}$ 会变成常量。

$P_{I_1}$ 和 $P_{I_2}$ 可以由 $P_{I_1,I_2}$ 计算得到：

$$
P_{I_1}(i)=\sum_kP_{I_1,I_2}(i,k)
$$

然后，互信息 Mutual information：

$$
MI_{I_1,I_2}=\sum_pmi_{I_1,I_2}(I_{1\mathbf{p}},I_{2\mathbf{p}})-----(8a) \\
mi_{I_1,I_2}(i,k)=h_{I_1}(i)+h_{I_2}(k)-h_{I_1,I_2}(i,k)-----(8b)
$$

于是导出 MI matching cost:

$$
C_{MI}(\mathbf{p},d)=-mi_{I_b,f_D(I_m)}(I_{b\mathbf{p}},I_{m\mathbf{q}})-----(9a) \\
\mathbf{q}=e_{bm}(\mathbf{p},d)-----(9b)
$$

由于需要先有 $f_D(I_m)$, 一般通过迭代的方式，先各一个随机估计，然后通过迭代进行更新。但是这样的话就会使运算量成倍增加。

这里提出了一个漂亮的解决方法，通过subsample，将图像长宽以及视差深度缩小为1/2，那样的话，运算量变为1/8。多次subsample的话就很小了，总的运算量为原来的：

$$
1+\frac{1}{2^3}+\frac{1}{4^3}+\frac{1}{8^3}+3\frac{1}{16^3}\approx1.14-----(10)
$$

运算量仅为一次完整迭代的 1.14 倍。


### 2.2 Cost Aggregation 代价聚合

对于视差图 $D$ 的能量 $E(D)$ 定义为:

$$
E(D)=\sum_\mathbf{p}(C(\mathbf{p},D_\mathbf{p})+\sum_{\mathbf{q}\in N_\mathbf{p}}P_1T[|D_\mathbf{p}-D_\mathbf{q}|=1] \\
+\sum_{\mathbf{q}\in N_\mathbf{p}}P_2T[|D_\mathbf{p}-D_\mathbf{q}|>1])-----(11)
$$

其中 $\mathbf{q}$ 为 $\mathbf{p}$ 的领域 $N_\mathbf{p}$ 中的点，$P_1$ 为 $D_\mathbf{p},D_\mathbf{q}$ 相差较小的惩罚常量，$P_2$ 为 $D_\mathbf{p},D_\mathbf{q}$ 相差较大的惩罚常量。 $P_2>=P_1$。

定义能量后，立体匹配问题就变成了寻找 $D$ 使得 $E(D)$ 尽可能小。


像素 $\mathbf{p}$ 对应视差 $d$ 时的代价 $L_\mathbf{r}^{'}(\mathbf{p},d)$ 沿着方向 $\mathbf{r}$ 传递过来为：
$$
L_\mathbf{r}^{'}(\mathbf{p},d)=C(\mathbf{p},d)+min(L_{\mathbf{r}}^{'}(\mathbf{p}-\mathbf{r},d),\\
L_\mathbf{r}^{'}(\mathbf{p}-\mathbf{r},d-1)+P_1,\\
L_\mathbf{r}^{'}(\mathbf{p}-\mathbf{r},d+1)+P_1,\\
\underset{i}{min}L_\mathbf{r}^{'}(\mathbf{p}-\mathbf{r},i)+P_2)-----(12)
$$

如果一直沿着路径计算，$L^{'}$ 会变得很大，因此，修改成：

$$
L_\mathbf{r}(\mathbf{p},d)=C(\mathbf{p},d)+min(L_{\mathbf{r}}(\mathbf{p}-\mathbf{r},d),\\
L_\mathbf{r}(\mathbf{p}-\mathbf{r},d-1)+P_1,\\
L_\mathbf{r}(\mathbf{p}-\mathbf{r},d+1)+P_1,\\
\underset{i}{min}L_\mathbf{r}(\mathbf{p}-\mathbf{r},i)+P_2)-\underset{k}{min}L_{\mathbf{r}}(\mathbf{p}-\mathbf{r},k)-----(13)
$$

这种修改并不会影响到路径以及视差的计算。

最终聚合后的代价为：

$$
S(\mathbf{p},d)=\sum_\mathbf{r}L_\mathbf{r}(\mathbf{p},d)-----(14)
$$

$r$ 为8或16个方向。

聚合示意图：

![](https://cdn.jsdelivr.net/gh/yjf0602/PicHost/img/sgm_aggregation.png)


### 2.3 Disparity Computation 视差计算

base image $I_b$ 对应的视差图 $D_b$ 使用与局部算法一样的方法确定视差 $d$，选取最小代价，$min_dS[\mathbf{p},d]$。对于 subpixel estimation, 使用二次曲线进行拟合。

match image $I_m$ 对应的视差图 $D_m$ 使用同样的方法得到。

最后根据左右视差图 $I_b,I_m$ 求最终视差：

$$
D_\mathbf{p}=\left\{\begin{matrix}
D_{b\mathbf{p}} & if |D_{b\mathbf{p}}-D_{m\mathbf{q}}|\leq1,\\ 
D_{inv} & otherwise.
\end{matrix}\right.-----(15a)\\
\mathbf{q}=e_{bm}(\mathbf{p},D_{b\mathbf{p}}).-----(15b)
$$

SGM 算法的总结示意图：

![](https://cdn.jsdelivr.net/gh/yjf0602/PicHost/img/sgm_SGM.png)


### 2.4 Multibaseline Matching

Todo(not really

### 2.5 Disparity refinement 视差优化

#### 2.5.1 Remove of Peaks

**segments**

#### 2.5.2 Intensity Consistent Disparity Selection

#### 2.5.3 Discontinuity Preserving Interpolation


### 2.6 Processing of Huge Images

### 2.7 Fusion of Disparity Images

