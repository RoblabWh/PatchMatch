#pragma once

#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

namespace dense {

float Variance(float *x, int n);

class NCCEstimator {
 public:
  NCCEstimator();
  void Push(float x, float y, float w);
  float Get();

 private:
  float sumx_, sumy_;
  float sumxx_, sumyy_, sumxy_;
  float sumw_;
};


void ApplyHomography(const cv::Matx33f &H, float x1, float y1, float *x2,
                     float *y2);

cv::Matx33d PlaneInducedHomography(const cv::Matx33d &K1, const cv::Matx33d &R1,
                                   const cv::Vec3d &t1, const cv::Matx33d &K2,
                                   const cv::Matx33d &R2, const cv::Vec3d &t2,
                                   const cv::Vec3d &v);

cv::Matx33f PlaneInducedHomographyBaked(const cv::Matx33d &K1inv,
                                        const cv::Matx33d &Q2,
                                        const cv::Vec3d &a2,
                                        const cv::Matx33d &K2,
                                        const cv::Vec3d &v);

cv::Vec3d Project(const cv::Vec3d &x, const cv::Matx33d &K,
                  const cv::Matx33d &R, const cv::Vec3d &t);

cv::Vec3d Backproject(double x, double y, double depth, const cv::Matx33d &K,
                      const cv::Matx33d &R, const cv::Vec3d &t);

float DepthOfPlaneBackprojection(double x, double y, const cv::Matx33d &K,
                                 const cv::Vec3d &plane);

cv::Vec3d PlaneFromDepthAndNormal(float x, float y, const cv::Matx33d &K,
                                  float depth, const cv::Vec3d &normal);

float UniformRand(float a, float b);

float DepthOfPlaneBackprojection_equirect(cv::Mat &Bearings_, int j, int i, const cv::Vec3d& plane);
cv::Vec3d Backproject_equirect(cv::Mat &Bearings_, int j, int i, float depth, const cv::Matx33d& rot_cw, const cv::Vec3d& trans_cw);
cv::Vec3d BearingFromImgPos_equirect(cv::Mat &Bearings_, int j, int i);
cv::Vec3d NormBearing_equirect(const cv::Vec3d &bearing);
cv::Vec3d Point3d_FromBearingDepth(float depth, const cv::Vec3d& bearing);
cv::Vec3d Point3d_FromPixAtDepth(cv::Mat &Bearings_, int j, int i, float depth);
cv::Vec3d PlanePosFromDepthAndNormal_equirect(cv::Mat &Bearings_, int j, int i, float depth, const cv::Vec3d &normal);
bool ReprojectEquirect(int cols, int rows, const cv::Matx33d& rot_cw, const cv::Vec3d& trans_cw, const cv::Vec3d& pos_w, float& x, float& y) ;
void ComputeBearingTable(cv::Mat &Bearings_);



struct DepthmapEstimatorResult {
  cv::Mat depth;
  cv::Mat plane;
  cv::Mat score;
  cv::Mat nghbr;
  cv::Mat patchs;
  cv::Matx33d R;
  cv::Vec3d t;
};

void PatchMatchUpdatePixel(DepthmapEstimatorResult *result, int i, int j,int adjacent[8][2], bool sample, std::mt19937 &rng_, 
std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int patch_size_);
void CheckPlaneCandidate(DepthmapEstimatorResult *result, int i, int j,const cv::Vec3d &plane, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int patch_size_);
void RandomInitialization_(DepthmapEstimatorResult *result, bool sample, int patch_size_, std::mt19937 &rng_, 
std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &mask, float min_depth_, float max_depth_, cv::Mat &image, std::vector<cv::Mat> &images_, std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_);

/* Red black patten passes */
void PatchMatchRedPass(DepthmapEstimatorResult *result, bool sample, int patch_size_, std::mt19937 &rng_, std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_);
void PatchMatchBlackPass(DepthmapEstimatorResult *result, bool sample, int patch_size_, std::mt19937 &rng_, std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_);
float ComputePlaneImageScoreUnoptimized_equirect(std::vector<cv::Mat> &images_, cv::Mat &Bearings, std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int cols, int rows, int i, int j, const cv::Vec3d &plane, int other, DepthmapEstimatorResult *result, int patch_size_);

void CheckPlaneImageCandidate(DepthmapEstimatorResult *result, int i, int j, const cv::Vec3d &plane,int nghbr, 
cv::Mat &image, std::vector<cv::Mat> &images_, cv::Mat &Bearings, std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int patch_size_);
void AssignPixel(DepthmapEstimatorResult *result, int i, int j,
                  const float depth, const cv::Vec3d &plane, const float score,
                  const int nghbr, int patchsize=0);
void ComputePlaneScore(int i, int j, const cv::Vec3d &plane,float *score, int *nghbr, DepthmapEstimatorResult *result, 
cv::Mat &image, std::vector<cv::Mat> &images_, cv::Mat &Bearings, std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int patch_size_);
float ComputePlaneImageScoreUnoptimized(int i, int j, const cv::Vec3d &plane,
                                        int other);
//float ComputePlaneImageScore(int i, int j, const cv::Vec3d &plane, int other, DepthmapEstimatorResult *result);
float BilateralWeight(float dcolor, float dx, float dy);
void PostProcess(cv::Mat &Bearings, DepthmapEstimatorResult *result, cv::Matx33d &R, cv::Vec3d &t, const float min_depth, const float max_depth, const float min_score);

void ComputeIgnoreMask(DepthmapEstimatorResult *result, std::vector<cv::Mat> &images_, std::vector<cv::Mat> &masks_, int patch_size_, float min_patch_variance_);
float PatchVariance(std::vector<cv::Mat> &images_, int i, int j, int patch_size_);

void GuessPlane(int i, int j, DepthmapEstimatorResult *result, std::mt19937 &rng_, std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_);
void PreInitSparsePCL(DepthmapEstimatorResult *result, std::vector<std::vector<cv::Vec3d>> &landmarks_, cv::Matx33d &R, cv::Vec3d &t, cv::Mat& mask);



class DepthmapEstimator {
 public:
  DepthmapEstimator();
  ~DepthmapEstimator();
  void AddView(cv::Matx33d &R, cv::Vec3d &t,cv::Mat &Img, cv::Mat &mask, std::vector<cv::Vec3d> &landmarks, double &maxd, double& mind);

  cv::Mat GetHeadImg();
  int PopHeadUnlessOne();
  void AssignMatrices(DepthmapEstimatorResult *result);
  void reset();
  int DebugShiftHead();
  bool IsReadyForCompute();
  bool SetMinImagesCompute(int min_images_compute);
  void SetDepthRange(double min_depth, double max_depth, int num_depth_planes);
  void SetPatchMatchIterations(int n);
  void SetPatchSize(int size);
  void SetMinPatchSD(float sd);
  void SetMinScore(double min_score);
  void ComputeBruteForce(DepthmapEstimatorResult *result);
  void ComputePatchMatch(DepthmapEstimatorResult *result);
  void ComputePatchMatchSample(DepthmapEstimatorResult *result);
  
  void PatchMatchForwardPass(DepthmapEstimatorResult *result, bool sample);
  void PatchMatchBackwardPass(DepthmapEstimatorResult *result, bool sample);

  void swapCenter();
  void MemorizeResult(DepthmapEstimatorResult *result);
  void PreInitResultWarp(
    DepthmapEstimatorResult *result, cv::Matx33d &R, cv::Vec3d &t, cv::Mat &Bearings, cv::Mat &mask, const int patchhalf);

  void RandomInitializationGPU(
    DepthmapEstimatorResult *result, const int patch_size_, const float min_depth_, const float max_depth_, 
    cv::Mat &img1, cv::Mat &img2, cv::Matx33d &R1, cv::Vec3d &t1, cv::Matx33d &R2, cv::Vec3d &t2);

  void ComputeBearingTableGPU(const int width, const int height);
  void PreInitSparsePCL2(
  DepthmapEstimatorResult *result, std::vector<std::vector<cv::Vec3d>> &landmarks_,
  cv::Matx33d &R, cv::Vec3d &t, cv::Mat& mask, cv::Mat &Bearings, const int patchhalf);


 private:
  double sum_time_;
  int count_time_;
  double min_score_;
  curandState* d_StatesMap_;
  float* d_Bearingmap_;
  float* d_Pano1_;
  float* d_Pano2_;
  float* d_Pano3_;
  cudaTextureObject_t* d_texs_;
  float* d_Depthmap_;
  float* d_Planemap_;
  float* d_Scoremap_;
  

  std::vector<cv::Mat> images_;
  std::vector<cv::Mat> masks_;
  //std::vector<cv::Matx33d> Ks_;
  std::vector<cv::Matx33d> Rs_;
  std::vector<cv::Vec3d> ts_;
  std::vector<double> maxds_;
  std::vector<double> minds_;
  std::vector<std::vector<cv::Vec3d>> landmarks_;
  
  int patch_size_;
  double min_depth_, max_depth_;
  int num_depth_planes_;
  int patchmatch_iterations_;
  float min_patch_variance_;
  std::mt19937 rng_;
  std::uniform_int_distribution<int> uni_;
  std::normal_distribution<float> unit_normal_;
  int min_images_compute_;
  int nDebugShifts_;
  cv::Mat Bearings_;

  cv::Matx33d result_R_;
  cv::Vec3d result_t_;
  cv::Mat* result_depth_;
  cv::Mat* result_plane_;
};

struct DepthmapCleanerResult {
  cv::Mat cleaned_depth;
  //cv::Mat cleaned_plane;
  //cv::Mat cleaned_score;
  cv::Matx33d R;
  cv::Vec3d t;
};

class DepthmapCleaner {
 public:
  DepthmapCleaner();
  void ComputeBearingTable();
  int PopHeadUnlessOne();
  int DebugShiftHead();
  void reset();
  void SetSameDepthThreshold(float t);
  void SetMinConsistentViews(int n);
  void AddView(DepthmapEstimatorResult* result); //cv::Matx33d &R,cv::Vec3d &t, cv::Mat &depth);
  void Clean(DepthmapCleanerResult* result); //cv::Mat &clean_depth);
  //DepthmapCleanerResult Get_result();
  bool IsReadyForCompute();
  void SetDepthQueueSize(int n);
  void swapCenter();
  void AssignMatrices(DepthmapCleanerResult *result);

 private:
  //std::vector<cv::Mat> depths_;
  //std::vector<cv::Matx33d> Rs_;
  //std::vector<cv::Vec3d> ts_;
  std::vector<DepthmapEstimatorResult*> depth_estimate_results_;
  cv::Mat Bearings_;
  float same_depth_threshold_;
  int min_consistent_views_;
  int nDebugShifts_;
  int nDepths_;
  DepthmapCleanerResult *result;
  
};

struct DepthmapPrunerResult {
  cv::Mat pruned_depth;
  //cv::Mat pruned_plane;
  //cv::Mat pruned_score;
  std::vector<cv::Vec3d> points3d;
  std::vector<cv::Vec3b> colors;
  std::vector<cv::Vec2i> keypts;
  cv::Matx33d R;
  cv::Vec3d t;
};

class DepthmapPruner {
 public:
  DepthmapPruner();
  void ComputeBearingTable();
  void SetSameDepthThreshold(float t);
  int DebugShiftHead();
  int PopHead();
  void reset();
  void AssignMatrices(DepthmapPrunerResult *result);
  void AddView(DepthmapCleanerResult* cleaned_result, cv::Mat &rgb); //cv::Matx33d &R,cv::Vec3d &t, cv::Mat &Depth,cv::Mat &Plane, cv::Mat &Img);
  /*
  void Prune(std::vector<cv::Vec3d> &merged_points,
             std::vector<cv::Vec3d> &merged_normals,
             std::vector<cv::Vec3d> &merged_colors,
             std::vector<cv::Vec2i> &merged_keypts);
  */
  void Prune(DepthmapPrunerResult* result);
  bool IsReadyForCompute();
  void SetMinViews(int min_views);
  void SetDepthQueueSize(int n);
  void swapCenter();

 private:

 std::vector<DepthmapCleanerResult*> cleaned_results_;
 DepthmapPrunerResult* result;

  //std::vector<cv::Mat> depths_;
  //std::vector<cv::Mat> planes_;
  std::vector<cv::Mat> images_;
  //std::vector<cv::Matx33d> Rs_;
  //std::vector<cv::Vec3d> ts_;
  cv::Mat Bearings_;
  float same_depth_threshold_;
  int min_views_;
  int nDebugShifts_;
  int nDepths_;
};

}  // namespace dense
