#include <opencv2/opencv.hpp>
#include "depthmap.h"
#include <random>
#include <exception>
#include <chrono>
#include <math.h>

#include <omp.h>

extern "C" void ComputePatchMatchPano(float* h_Scoremap, float* h_Depthmap, float* h_Planemap, float* h_Bearingmap, int width, int height, cv::Mat &pano1, cv::Mat &pano2, float* R12, float* t12);
extern "C" void ComputeBearingsEquirect(float* d_Bearingmap, const int width, const int height);
extern "C" void ComputeRandomSeed(curandState* d_StatesMap, const int width, const int height);
extern "C" void ComputeRandomPlanemapDepthmap(float* d_Depthmap, float* d_Planemap,float*  d_Bearingmap, int width, int height, float min_depth, float max_depth, curandState* d_StatesMap);
extern "C" void InitTextureImages(const int width, const int height, cv::Mat &Img1, cv::Mat &Img2, cv::Mat &Img3, float* d_img1, float* d_img2, float* d_img3, cudaTextureObject_t texs[]);
extern "C" void ScorePlaneDepth(const int patchhalf, float* d_Scoremap, float* d_Depthmap, float* d_Planemap, float* d_Bearingmap, cudaTextureObject_t* d_texs, int width, int height);
extern "C" void RandomInitialization(const int patchhalf, float* d_Scoremap, float* d_Depthmap, float* d_Planemap, float* d_Bearingmap,
		cudaTextureObject_t* d_texs, int width, int height, float min_depth, float max_depth, curandState* d_StatesMap);
extern "C" void InitConstantMem(float* R12, float* t12, float* R13, float* t13, float* R1_inv, float* t1_inv);
extern "C" void PatchMatchRedBlackPass(
  const int patchhalf, int nIterations, float* d_Scoremap, float* d_Depthmap, float* d_Planemap,
  float* d_Bearingmap, cudaTextureObject_t* d_texs, int width, int height, curandState* d_StatesMap);



namespace dense {

static const double z_epsilon = 1e-8;


bool IsInsideImage(const cv::Mat &image, int i, int j) {
  return i >= 0 && i < image.rows && j >= 0 && j < image.cols;
}

template <typename T>
float LinearInterpolation(const cv::Mat &image, float y, float x) {
  if (x < 0.0f || x >= image.cols - 1 || y < 0.0f || y >= image.rows - 1) {
    return 0.0f;
  }
  int ix = int(x);
  int iy = int(y);
  float dx = x - ix;
  float dy = y - iy;
  float im00 = image.at<T>(iy, ix);
  float im01 = image.at<T>(iy, ix + 1);
  float im10 = image.at<T>(iy + 1, ix);
  float im11 = image.at<T>(iy + 1, ix + 1);
  float im0 = (1 - dx) * im00 + dx * im01;
  float im1 = (1 - dx) * im10 + dx * im11;
  return (1 - dy) * im0 + dy * im1;
}

float Variance(float *x, int n) {
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
  }
  float mean = sum / n;

  float sum2 = 0;
  for (int i = 0; i < n; ++i) {
    sum2 += (x[i] - mean) * (x[i] - mean);
  }
  return sum2 / n;
}

NCCEstimator::NCCEstimator()
    : sumx_(0), sumy_(0), sumxx_(0), sumyy_(0), sumxy_(0), sumw_(0) {}

void NCCEstimator::Push(float x, float y, float w) {
  sumx_ += w * x;
  sumy_ += w * y;
  sumxx_ += w * x * x;
  sumyy_ += w * y * y;
  sumxy_ += w * x * y;
  sumw_ += w;
}

float NCCEstimator::Get() {
  if (sumw_ == 0.0) {
    return -1;
  }
  float meanx = sumx_ / sumw_;
  float meany = sumy_ / sumw_;
  float meanxx = sumxx_ / sumw_;
  float meanyy = sumyy_ / sumw_;
  float meanxy = sumxy_ / sumw_;
  float varx = meanxx - meanx * meanx;
  float vary = meanyy - meany * meany;
  if (varx < 0.1 || vary < 0.1) {
    return -1;
  } else {
    return (meanxy - meanx * meany) / sqrt(varx * vary);
  }
}


float DepthOfPlaneBackprojection_equirect(cv::Mat &Bearings_, int j, int i, const cv::Vec3d& plane) {

  float depth = 0;
  cv::Vec3d bearing = Bearings_.at<cv::Vec3d>(i,j);
  float denom = plane.dot(bearing);
  depth = 1.0f / std::max(std::abs(denom),1e-6f);
  return depth;
}

cv::Vec3d Backproject_equirect(cv::Mat &Bearings_, int j, int i, float depth, const cv::Matx33d& rot_cw, const cv::Vec3d& trans_cw) {
  cv::Vec3d point3d = depth * Bearings_.at<cv::Vec3d>(i,j);
  cv::Vec3d pos_w = rot_cw.t() * (point3d - trans_cw);

  return pos_w;
}

cv::Vec3d BearingFromImgPos_equirect(int cols, int rows, float x, float y) {

  // convert to unit polar coordinates
  const float lon = (x / cols - 0.5) * (2 * M_PI);
  const float lat = -(y / rows - 0.5) * M_PI;

  // convert to equirectangular coordinates
  const float cos_lat = cosf(lat);
  const float cos_lon = cosf(lon);
  const float sin_lon = sinf(lon);
  const float sin_lat = sinf(lat);

  cv::Vec3d bearing(cos_lat * sin_lon, -sin_lat, cos_lat * cos_lon);

  return bearing;
}

cv::Vec3d PlanePosFromDepthAndNormal_equirect(cv::Mat &Bearings_, int j, int i, float depth, const cv::Vec3d &normal) {

  cv::Vec3d point = depth * Bearings_.at<cv::Vec3d>(i,j);
  double denom = normal.dot(point);
  if (denom < 1e-5 && denom >= 0) {
    denom = 1e-5;
  }
  else if (denom > -1e-5 && denom < 0) {
    denom = -1e-5;
  }
  return normal / denom;
}

bool ReprojectEquirect(int cols, int rows, const cv::Matx33d& rot_cw, const cv::Vec3d& trans_cw, const cv::Vec3d& pos_w, float& x, float& y) {
    // convert to camera-coordinates
    const cv::Vec3d bearingtmp = ((rot_cw * pos_w) + trans_cw);
    double depth = cv::norm(bearingtmp);
    if (depth < 1e-4) {
      return true; //too near
    }
    const cv::Vec3d bearing = bearingtmp / depth;

    // convert to unit polar coordinates
    const float latitude = -asinf(bearing(1));
    const float longitude = atan2f(bearing(0), bearing(2));

    // convert to pixel image coordinated
    x = cols * (0.5 + longitude / (2.0 * M_PI));
    y = rows * (0.5 - latitude / M_PI);
    
    return false;
}

cv::Vec3d PosPatchPartOnPlane(cv::Matx33d &R, cv::Vec3d &t, cv::Mat &Bearings_, int i, int j, const cv::Vec3d &plane) {

  //cv::Vec3d bearing = BearingFromImgPos_equirect(cols,rows, x, y);
  cv::Vec3d norm_bearing = Bearings_.at<cv::Vec3d>(i,j);
  float scale = plane.dot(norm_bearing);
  if (scale < 1e-6 && scale >= 0) {
    scale = 1e-6;
  }
  else if (scale > -1e-6 && scale < 0) {
    scale = -1e-6;
  }
  cv::Vec3d posOnPlane_c = norm_bearing / scale;
  cv::Vec3d posOnPlane_w = R.t() * (posOnPlane_c - t);
  return posOnPlane_w;
}

float UniformRand(float a, float b) {
  return a + (b - a) * float(rand()) / RAND_MAX;
}

DepthmapEstimator::DepthmapEstimator()
    : patch_size_(7),
      min_depth_(0),
      max_depth_(0),
      num_depth_planes_(50),
      patchmatch_iterations_(3),
      min_patch_variance_(5 * 5),
      rng_{std::random_device{}()},
      uni_(0, 0),
      unit_normal_(0, 1),
      min_images_compute_(2), nDebugShifts_(0), min_score_(0) {

        d_StatesMap_ = NULL;
        d_Bearingmap_ = NULL;


        
 
      }

DepthmapEstimator::~DepthmapEstimator() {

  if (d_StatesMap_ != NULL) {
    cudaFree(d_StatesMap_);
    d_StatesMap_ = NULL;
  }

  if (d_Bearingmap_ != NULL) {
    cudaFree(d_Bearingmap_);
    d_Bearingmap_ = NULL;
  }
  
}

//, std::vector<double> &landmarks
void DepthmapEstimator::AddView(cv::Matx33d &R,
                                cv::Vec3d &t, cv::Mat &Img,
                                cv::Mat &mask, std::vector<cv::Vec3d> &landmarks, double &maxd, double& mind) {
  //Ks_.emplace_back(K);
  Rs_.emplace_back(cv::Matx33d(R));
  ts_.emplace_back(cv::Vec3d(t));
  images_.emplace_back(Img.clone());
  masks_.emplace_back(mask.clone());
  std::vector<cv::Vec3d> points = landmarks;
  landmarks_.emplace_back(points);

  maxds_.emplace_back(maxd);
  minds_.emplace_back(mind);
  std::size_t size = images_.size();
  int a = (size > 1) ? 1 : 0;
  int b = (size > 1) ? size - 1 : 0;
  uni_.param(std::uniform_int_distribution<int>::param_type(a, b));

}
cv::Mat DepthmapEstimator::GetHeadImg() {
  return *images_.begin();
}

int DepthmapEstimator::PopHeadUnlessOne() {

  int nViews = images_.size();
  if (nViews > 1) {
    Rs_.erase(Rs_.begin());
    ts_.erase(ts_.begin());
    images_.erase(images_.begin());
    masks_.erase(masks_.begin());
    maxds_.erase(maxds_.begin());
    minds_.erase(minds_.begin());
    landmarks_.erase(landmarks_.begin());
  }
  return images_.size();
}

void DepthmapEstimator::reset() {

  Rs_.clear();
  ts_.clear();

  if(result_depth_) {
    delete result_depth_;
    result_depth_ = nullptr;
  }
  if(result_plane_) {
    delete result_plane_;
    result_plane_ = nullptr;
  }
  result_depth_ = new cv::Mat(images_[0].rows, images_[0].cols, CV_32F, 0.0f);
  result_plane_ = new cv::Mat(images_[0].rows, images_[0].cols, CV_64FC3, 0.0f);

  images_.clear();
  masks_.clear();
  minds_.clear();
  maxds_.clear();
  nDebugShifts_ = 0;
  landmarks_.clear();

  
  result_R_ = cv::Matx33d(0,0,0,0,0,0,0,0,0);
  result_t_ = cv::Vec3d(0,0,0);
}

int DepthmapEstimator::DebugShiftHead() {

  
  if (nDebugShifts_ < images_.size()) {
    cv::Matx33d R0(Rs_[0]);
    Rs_.erase(Rs_.begin());
    Rs_.emplace_back(R0);

    cv::Vec3d t0(ts_[0]);
    ts_.erase(ts_.begin());
    ts_.emplace_back(t0);
    
    //cv::Mat image0 = images_[0].clone();
    images_.emplace_back(images_[0].clone());
    images_.erase(images_.begin());
    
    double maxd = maxds_[0];
    double mind = minds_[0];
    maxds_.erase(maxds_.begin());
    minds_.erase(minds_.begin());
    maxds_.emplace_back(maxd);
    minds_.emplace_back(mind);

    nDebugShifts_ += 1;
  }
  return nDebugShifts_;
}

bool DepthmapEstimator::IsReadyForCompute() {
  return images_.size() >= min_images_compute_ && nDebugShifts_ < images_.size();
}

bool DepthmapEstimator::SetMinImagesCompute(int min_images_compute) {

  bool bmin = min_images_compute >= 2;
  min_images_compute_ = bmin ? min_images_compute : 2;
  return bmin;
}

void DepthmapEstimator::SetDepthRange(double min_depth, double max_depth,
                                      int num_depth_planes) {
  min_depth_ = min_depth;
  max_depth_ = max_depth;
  num_depth_planes_ = num_depth_planes;
}

void DepthmapEstimator::SetPatchMatchIterations(int n) {
  patchmatch_iterations_ = n;
}

void DepthmapEstimator::SetPatchSize(int size) { patch_size_ = size; }

void DepthmapEstimator::SetMinPatchSD(float sd) {
  min_patch_variance_ = sd * sd;
}

void DepthmapEstimator::SetMinScore(double min_score) { min_score_=min_score; }

void DepthmapEstimator::ComputeBruteForce(DepthmapEstimatorResult *result) {
}

void ComputeBearingTable(cv::Mat &Bearings_) {

  
  #pragma omp parallel for //num_threads(4)
  for (int i =0; i< Bearings_.rows; i++) {
    //std::cout << "in thread: running bearing row " << std::to_string(i) << std::endl;
    for (int j = 0; j < Bearings_.cols; j++) {
      
      //cv::Vec3d bearing =  NormBearing_equirect(BearingFromImgPos_equirect(Bearings_.cols, Bearings_.rows, j, i));
      cv::Vec3d bearing =  BearingFromImgPos_equirect(Bearings_.cols, Bearings_.rows, j, i);
      Bearings_.at<cv::Vec3d>(i, j) = bearing / cv::norm(bearing);
    }
  }
}


void DepthmapEstimator::PreInitSparsePCL2(
  DepthmapEstimatorResult *result, std::vector<std::vector<cv::Vec3d>> &landmarks_,
  cv::Matx33d &R, cv::Vec3d &t, cv::Mat& mask, cv::Mat &Bearings, const int patchhalf) {

  #pragma omp parallel for
  for (int i = 0; i< landmarks_[0].size(); i++) {
    float x, y, depth;
    int x0, y0;
    cv::Vec3d normal;
    cv::Vec3d bearing;
    cv::Vec3d plane;
    cv::Vec3d point_w = landmarks_[0][i];
    cv::Vec3d point_c = R * point_w + t;
    int width, height;
    
    bool tooNear = ReprojectEquirect(result->depth.cols, result->depth.rows, R, t, point_w, x, y);
    if (tooNear) {
      continue;
    }

    x0 = int(x+0.5);
    y0 = int(y+0.5);

    if (!mask.at<unsigned char>(y0, x0)) {
      continue;
    }

    //if (result->depth.at<float>(y0,x0) > 0) {
    //  continue; //already known through warping
    //}

    depth = cv::norm(point_c);

    //assume normal is inverse z-direction
    bearing = Bearings.at<cv::Vec3d>(y0, x0);
    normal = bearing;
    normal *= -1;

    //get plane form normal an depth
    plane = PlanePosFromDepthAndNormal_equirect(Bearings, x0, y0, depth, normal);
    float score;
    int nghbr;
    ComputePlaneScore(y0, x0, plane, &score, &nghbr, result, images_[0], images_, Bearings_, Rs_, ts_,  patch_size_);
    if (score > result->score.at<float>(y0, x0)) {
      result->plane.at<cv::Vec3d>(y0, x0) = plane;
      result->depth.at<float>(y0, x0) = depth;
      result->score.at<float>(y0, x0) = score;
    }

    width = result->depth.cols;
    height = result->depth.rows;

    //propagate to patch neighboors
    /*
    for (int dy = -patchhalf; dy <= patchhalf; dy+=1) {
      for (int dx = -patchhalf; dx <= patchhalf; dx+=1) {
        int y = y0+dy;
        int x = x0+dx;
        if (y < 0 || x < 0 || y >= height || x >= width) {
          continue;
        }
        if (!mask.at<unsigned char>(y, x)) {
          continue;
        }

        depth = DepthOfPlaneBackprojection_equirect(Bearings, x, y, plane);
        result->depth.at<float>(y,x) = depth;
        result->plane.at<cv::Vec3d>(y,x) = plane;
      }
    }*/
  }
}

void DepthmapEstimator::PreInitResultWarp(
  DepthmapEstimatorResult *result, cv::Matx33d &R, cv::Vec3d &t, cv::Mat &Bearings, cv::Mat &mask, const int patchhalf) {
  
  if (!result_depth_ || !result_plane_) {
    return;
  }

  float x, y;

  #pragma omp parallel for
  for (int r = 0; r < result_depth_->rows; r++) {
    for (int c = 0; c < result_depth_->cols; c++) {
      if (!mask.at<unsigned char>(r,c)) {
        continue;
      }
      cv::Vec3d plane_c1 = result_plane_->at<cv::Vec3d>(r,c);
      float depth = result_depth_->at<float>(r,c);
      if (depth < 1e-4) {
        continue;
      }
      
      cv::Vec3d point_w = PosPatchPartOnPlane(result_R_, result_t_, Bearings, r, c, plane_c1);
      bool tooNear = ReprojectEquirect(result_depth_->cols, result_depth_->rows, R, t, point_w, x, y);
      if (tooNear) {
        continue;
      }
      int i = int(y+0.5);
      int j = int(x+0.5);
      if (i < 0 || j < 0 || i >= result->depth.rows || j >= result->depth.cols) {
        continue;
      }
      if (!mask.at<unsigned char>(i, j)) {
        continue;
      }

      cv::Vec3d point_c = R * point_w + t;
      depth = cv::norm(point_c);
      
      //cv::Vec3d plane = R * plane_c1 + t;
      cv::Vec3d normal = result_plane_->at<cv::Vec3d>(r, c);
      normal = normal / cv::norm(normal);
      //cv::Vec3d normal = Bearings.at<cv::Vec3d>(i, j);
      //normal *= -1;

      //get plane form normal an depth
      float score;
      int nghbr;
      cv::Vec3d plane = PlanePosFromDepthAndNormal_equirect(Bearings, j, i, depth, normal);
      ComputePlaneScore(i, j, plane, &score, &nghbr, result, images_[0], images_, Bearings_, Rs_, ts_,  patch_size_);
      if (score > result->score.at<float>(i,j)) {
        result->plane.at<cv::Vec3d>(i,j) = plane;
        result->depth.at<float>(i,j) = depth;
        result->score.at<float>(i,j) = score;
      }
      
      /*
      //propagate to patch neighboors
      for (int dr = -patchhalf; dr <= patchhalf; dr+=1) {
        for (int dc = -patchhalf; dc <= patchhalf; dc+=1) {
          int i_N = i+dr;
          int j_N = j+dc;
          float depth_N;
          if (i_N < 0 || j_N < 0 || i_N >= result->depth.rows || j_N >= result->depth.cols) {
            continue;
          }
          if (!mask.at<unsigned char>(i_N, j_N)) {
            continue;
          }
          if (result->depth.at<float>(i_N, j_N) > 0) {
            continue;
          }

          depth_N = DepthOfPlaneBackprojection_equirect(Bearings, j_N, i_N, plane);
          if (depth > 1e-4) { 
            result->depth.at<float>(i_N, j_N) = depth_N;
            result->plane.at<cv::Vec3d>(i_N, j_N) = plane;
          }
          
        }
      }*/
    }
  }
}

void DepthmapEstimator::ComputeBearingTableGPU(const int width, const int height) {

  float* h_Bearingmap;
  size_t size_bearings = 3 * width * height * sizeof(float);
  cudaMalloc((void**)&d_Bearingmap_, size_bearings);
  h_Bearingmap = (float*) malloc(size_bearings);
  ComputeBearingsEquirect(d_Bearingmap_, width, height);
  cudaMemcpy(h_Bearingmap, d_Bearingmap_, size_bearings, cudaMemcpyDeviceToHost);
  for (int i =0; i< Bearings_.rows; i++) {
    for (int j = 0; j < Bearings_.cols; j++) {
      double bx, by, bz;
      bx = double(h_Bearingmap[i*3*width + 3*j]);
      by = double(h_Bearingmap[i*3*width + 3*j+1]);
      bz = double(h_Bearingmap[i*3*width + 3*j+2]);
      cv::Vec3d bearing(bx,by,bz);
      Bearings_.at<cv::Vec3d>(i, j) = bearing;
    }
  }

  free(h_Bearingmap);
  h_Bearingmap = NULL;
}

void DepthmapEstimator::ComputePatchMatch(DepthmapEstimatorResult *result) {
  AssignMatrices(result);
  
  //Bearings_ = cv::Mat(320, 320, CV_64FC3, 0.0f);
  //ComputeBearingTable(Bearings_);
  //ComputeIgnoreMask(result, images_, masks_, patch_size_, min_patch_variance_);
  swapCenter();
  int center_id = images_.size()/2;
  max_depth_ = maxds_[0];
  min_depth_ = minds_[0];
  int width = images_[0].cols;
  int height = images_[0].rows;
  
  Bearings_ = cv::Mat(height, width, CV_64FC3, 0.0f);
  ComputeBearingTableGPU(width, height);

  size_t size_states = width * height * sizeof(curandState);
  cudaMalloc((void**)&d_StatesMap_, size_states);
  ComputeRandomSeed(d_StatesMap_, width, height);

  cudaTextureObject_t texs[3];

  float* h_Depthmap;
  float* h_Planemap;
  float* h_Scoremap;
  cv::Mat pano1_f;
  cv::Mat pano2_f;
  cv::Mat pano3_f;
  images_[0].convertTo(pano1_f, CV_32F);
  images_[1].convertTo(pano2_f, CV_32F);
  images_[2].convertTo(pano3_f, CV_32F);

  InitTextureImages(width, height, pano1_f, pano2_f, pano3_f, d_Pano1_, d_Pano2_, d_Pano3_, texs);
  cudaMalloc((void**)&d_texs_,sizeof(cudaTextureObject_t)*3);
  cudaMemcpy(d_texs_,texs,sizeof(cudaTextureObject_t)*3,cudaMemcpyHostToDevice);

  float* R12 = (float*) malloc(9*sizeof(float));
  float* t12 = (float*) malloc(3*sizeof(float));
  float* R13 = (float*) malloc(9*sizeof(float));
  float* t13 = (float*) malloc(3*sizeof(float));

  float* R1w = (float*) malloc(9*sizeof(float));
  float* t1w = (float*) malloc(3*sizeof(float));

  cv::Mat P1 = cv::Mat::eye(4,4,CV_64F);
  cv::Mat P2 = cv::Mat::eye(4,4,CV_64F);
  cv::Mat P3 = cv::Mat::eye(4,4,CV_64F);
  cv::Mat P12;
  cv::Mat P13;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      P1.at<double>(i,j) = Rs_[0](i,j);
      P2.at<double>(i,j) = Rs_[1](i,j);
      P3.at<double>(i,j) = Rs_[2](i,j);
    }
  }

  for (int i = 0; i < 3; i++) {
    P1.at<double>(i,3) = ts_[0](i);
    P2.at<double>(i,3) = ts_[1](i);
    P3.at<double>(i,3) = ts_[2](i);
  }

  cv::Mat P1_inv = cv::Mat::eye(4,4,CV_64F);
  cv::Mat R1_inv = cv::Mat::eye(3,3,CV_64F);
  
  
  for (int y = 0; y< 3; y++) {
      for (int x = 0; x < 3; x++) {
          R1_inv.at<double>(y,x) = P1.at<double>(x,y);
          
      }
  }
  cv::Matx33d R1_inv2((double*)R1_inv.ptr());
  cv::Vec3d t1_inv = -R1_inv2 * ts_[0];
  for (int y = 0; y < 4; y++) {
      for (int x = 0; x < 4; x++) {
          if (y < 3 && x < 3) {
            R1w[y*3+x]=float(P1.at<double>(x,y));
            P1_inv.at<double>(y,x) = R1_inv.at<double>(y,x);
          }
          else if (x == 3 && y < 3) {
            P1_inv.at<double>(y,x) = t1_inv(y);
            t1w[y] = float(t1_inv(y));
          }
          
      }
  }
  
  
  //P12 = P1_inv * P2;
  P12 = P2;
  cv::Mat P12_=P12/P12.at<double>(3,3);

  //P13 = P1_inv * P3;
  P13 = P3;
  cv::Mat P13_=P13/P13.at<double>(3,3);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R12[i*3 + j] = float(P12_.at<double>(i,j));
      R13[i*3 + j] = float(P13_.at<double>(i,j));
    }
  }

  for (int i = 0; i < 3; i++) {
    t12[i] = float(P12_.at<double>(i,3));
    t13[i] = float(P13_.at<double>(i,3));
  }

  
  size_t size_depths = width * height * sizeof(float);
  size_t size_planes = 3 * width * height * sizeof(float);
  cudaMalloc((void**)&d_Depthmap_, size_depths);
  cudaMalloc((void**)&d_Scoremap_, size_depths);
  cudaMalloc((void**)&d_Planemap_, size_planes);
  cudaMemset(d_Scoremap_, -1.0f, size_depths);
  cudaMemset(d_Depthmap_, 0.0f, size_depths);
  cudaMemset(d_Planemap_, 0.0f, size_planes);
  h_Depthmap = (float*) malloc(size_depths);
  h_Scoremap = (float*) malloc(size_depths);
  h_Planemap = (float*) malloc(size_planes);

  //InitConstantMem(R12,t12, R13, t13);
  InitConstantMem(R12,t12, R13, t13, R1w, t1w);
  RandomInitialization(patch_size_-2, d_Scoremap_, d_Depthmap_, d_Planemap_, d_Bearingmap_, d_texs_, width, height, min_depth_, max_depth_, d_StatesMap_);
  
  
  cudaMemcpy(h_Planemap, d_Planemap_, size_planes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Depthmap, d_Depthmap_, size_depths, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Scoremap, d_Scoremap_, size_depths, cudaMemcpyDeviceToHost);
  
  for (int i=0; i < result->depth.rows; i++) {
    for (int j=0; j < result->depth.cols; j++) {
      if (!masks_[0].at<unsigned char>(i,j)) {
        continue;
      }
      double px, py, pz;
      px = double(h_Planemap[i*3*width + 3*j]);
      py = double(h_Planemap[i*3*width + 3*j+1]);
      pz = double(h_Planemap[i*3*width + 3*j+2]);
      cv::Vec3d plane(px,py,pz);
      result->plane.at<cv::Vec3d>(i, j) = plane;
      result->depth.at<float>(i,j) = h_Depthmap[i*width + j];
      result->score.at<float>(i,j) = h_Scoremap[i*width + j];
    }
  }
  
  
  PreInitSparsePCL2(result, landmarks_, Rs_[0], ts_[0], masks_[0], Bearings_, patch_size_/2);
  PreInitResultWarp(result, Rs_[0], ts_[0], Bearings_, masks_[0], patch_size_/2);
  
  
  for (int i=0; i < result->depth.rows; i++) {
    for (int j=0; j < result->depth.cols; j++) {
      if (!masks_[0].at<unsigned char>(i,j)) {
        continue;
      }
      double px, py, pz;
      float depth;
      float score;
      cv::Vec3d plane = result->plane.at<cv::Vec3d>(i, j);
      px = plane(0);
      py = plane(1);
      pz = plane(2);
      h_Planemap[i*3*width + 3*j] = px;
      h_Planemap[i*3*width + 3*j + 1] = py;
      h_Planemap[i*3*width + 3*j + 2] = pz;
      
      depth = result->depth.at<float>(i,j);
      score = result->score.at<float>(i,j);
      h_Depthmap[i*width+j] = depth;
      h_Scoremap[i*width+j] = score;
    }
  }

  cudaMemcpy(d_Planemap_, h_Planemap, size_planes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Depthmap_, h_Depthmap, size_depths, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Scoremap_, h_Scoremap,  size_depths, cudaMemcpyHostToDevice);


  const auto tp_1 = std::chrono::steady_clock::now();
  
  
  PatchMatchRedBlackPass(patch_size_-2, patchmatch_iterations_, d_Scoremap_, d_Depthmap_, d_Planemap_, d_Bearingmap_, d_texs_, width, height, d_StatesMap_); //patch_size_-2

  cudaMemcpy(h_Planemap, d_Planemap_, size_planes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Depthmap, d_Depthmap_, size_depths, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Scoremap, d_Scoremap_, size_depths, cudaMemcpyDeviceToHost);

  for (int i=patch_size_-2; i < result->depth.rows-patch_size_-2; i++) {
    for (int j=patch_size_-2; j < result->depth.cols-patch_size_-2; j++) {
      if (!masks_[0].at<unsigned char>(i,j)) {
        continue;
      }
      double px, py, pz;
      float depth;
      float score;
      bool low_variance = PatchVariance(images_, i, j, patch_size_) < min_patch_variance_;
      if (low_variance) {
        px=py=pz=0.0;
        depth = score = 0.0f;
      }
      else {
        px = double(h_Planemap[i*3*width + 3*j]);
        py = double(h_Planemap[i*3*width + 3*j+1]);
        pz = double(h_Planemap[i*3*width + 3*j+2]);

        depth = h_Depthmap[i*width + j];
        score = h_Scoremap[i*width + j];
      }
      
      
      cv::Vec3d plane(px,py,pz);
      result->plane.at<cv::Vec3d>(i, j) = plane;
      result->depth.at<float>(i,j) = depth;
      result->score.at<float>(i,j) = score;
      result->nghbr.at<int>(i, j) = 1;
    }
  }
  
  
  const auto tp_2 = std::chrono::steady_clock::now();
  const auto tp_delta = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
  //std::cout << "time elapsed here: " << std::to_string(tp_delta) << std::endl;
  
  PostProcess(Bearings_, result, Rs_[0], ts_[0], min_depth_, max_depth_, min_score_);
  MemorizeResult(result);

  
  free(R12);
  free(t12);
  free(R13);
  free(t13);
  free(R1w);
  free(t1w);
  
  cudaFree(d_texs_);
  d_texs_ = NULL;
  cudaFree(d_Depthmap_);
  d_Depthmap_ = NULL;
  cudaFree(d_Planemap_);
  d_Planemap_ = NULL;
  cudaFree(d_Scoremap_);
  d_Scoremap_ = NULL;
  cudaFree(d_StatesMap_);
  cudaFree(d_Bearingmap_);

  cudaFree(d_Pano1_);
  cudaFree(d_Pano2_);
  cudaFree(d_Pano3_);

  cudaDestroyTextureObject(texs[0]);
  cudaDestroyTextureObject(texs[1]);
  cudaDestroyTextureObject(texs[2]);
  cudaFree(d_texs_);

  free(h_Depthmap);
  free(h_Scoremap);
  free(h_Planemap);
  
  
  swapCenter();

  
}

void DepthmapEstimator::MemorizeResult(DepthmapEstimatorResult *result) {

  if (!result_depth_) {
    result_depth_ = new cv::Mat(images_[0].rows, images_[0].cols, CV_32F, 0.0f);
  }
  if (!result_plane_) {
    result_plane_ = new cv::Mat(images_[0].rows, images_[0].cols, CV_64FC3, 0.0f);
  }

  //#pragma omp parallel for
  for (int r = 0; r < result->depth.rows; r++) {
    for (int c = 0; c < result->depth.cols; c++) {
      cv::Vec3d plane_c = result->plane.at<cv::Vec3d>(r,c);
      //cv::Vec3d plane_w = result->R.t() * plane_c - result->R.t() * result->t;
      result_depth_->at<float>(r,c) = result->depth.at<float>(r,c);
      result_plane_->at<cv::Vec3d>(r,c) = plane_c;
      
    }
  }
  result_R_ = result->R;
  result_t_ = result->t;
}

void DepthmapEstimator::ComputePatchMatchSample(
    DepthmapEstimatorResult *result) {
}

void DepthmapEstimator::swapCenter() {

  int center_id = images_.size()/2;

  std::swap(Rs_[0],Rs_[center_id]);
  std::swap(ts_[0],ts_[center_id]);
  std::swap(images_[0],images_[center_id]);
  std::swap(maxds_[0],maxds_[center_id]);
  std::swap(minds_[0],minds_[center_id]);
  std::swap(landmarks_[0],landmarks_[center_id]);
}

void DepthmapEstimator::AssignMatrices(DepthmapEstimatorResult *result) {
  result->depth = cv::Mat(images_[0].rows, images_[0].cols, CV_32F, 0.0f);
  result->plane = cv::Mat(images_[0].rows, images_[0].cols, CV_64FC3, 0.0f);
  result->score = cv::Mat(images_[0].rows, images_[0].cols, CV_32F, -1.0f);
  result->nghbr =
      cv::Mat(images_[0].rows, images_[0].cols, CV_32S, cv::Scalar(0));
  result->R = Rs_[0];
  result->t = ts_[0];
}

void GuessPlane(int i, int j, DepthmapEstimatorResult *result, std::mt19937 &rng_, std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int patch_size_) {

    bool test = true;
    if (test) {

      cv::Vec3d current_plane = result->plane.at<cv::Vec3d>(i, j);
      CheckPlaneCandidate(result, i, j, current_plane, Bearings_, image, images_, Rs_, ts_, patch_size_);
      float depth_range = 0.005;
      float normal_range = 1.0;
      for (int k = 0; k < 12; ++k) {
          float depth = result->depth.at<float>(i, j);
          if (k>6) {
              depth = depth * exp(depth_range * unit_normal_(rng_));
          }
          
          cv::Vec3d normal(current_plane(0) + normal_range * unit_normal_(rng_),
                          current_plane(1) + normal_range * unit_normal_(rng_),
                          current_plane(2) + normal_range * unit_normal_(rng_));

          cv::Vec3d plane = PlanePosFromDepthAndNormal_equirect(Bearings_, j, i, depth, normal);
          CheckPlaneCandidate(result, i, j, plane, Bearings_, image,images_,Rs_,ts_,  patch_size_);

          if (k>6) {
            normal_range *= 0.5;
            depth_range *= 0.1;
          }
      }
    }
}

void DepthmapEstimator::RandomInitializationGPU(
    DepthmapEstimatorResult *result, const int patch_size_, const float min_depth, const float max_depth, 
    cv::Mat &img1, cv::Mat &img2, cv::Matx33d &R1, cv::Vec3d &t1, cv::Matx33d &R2, cv::Vec3d &t2) {

      
    }

void RandomInitialization_(DepthmapEstimatorResult *result, bool sample, int patch_size_, std::mt19937 &rng_,
std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &mask, float min_depth_, float max_depth_, cv::Mat &image, std::vector<cv::Mat> &images_,  std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_) {
  int hpz = (patch_size_ - 2);/// 2;
  try {
    #pragma omp parallel for //num_threads(8)
    for (int i = hpz; i < result->depth.rows - hpz; i++) {
      for (int j = hpz; j < result->depth.cols - hpz; j++) {

        if (!mask.at<unsigned char>(i, j)) {
          continue;
        }
        bool guess = true;
        if (result->depth.at<float>(i,j) > 0 && guess) {
            GuessPlane(i,j,result, rng_, unit_normal_, Bearings_, image,images_,Rs_,ts_, patch_size_);
        }
        else {
          /*
          int nghbr;
          float score;
          float depth = result->depth.at<float>(i,j);
          cv::Vec3d plane = result->plane.at<cv::Vec3d>(i,j);
          ComputePlaneScore(i, j, plane, &score, &nghbr, result, image, images_, Bearings_, Rs_, ts_,  patch_size_);
          AssignPixel(result, i, j, depth, plane, score, nghbr);
          */

          
          int nghbr;
          float score;
          float depth = exp(UniformRand(log(min_depth_), log(max_depth_)));
          //cv::Vec3d normal(0,0,-1);
          //cv::Vec3d plane = PlanePosFromDepthAndNormal_equirect(Bearings_, j, i, depth, normal);
          //ComputePlaneScore(i, j, plane, &score, &nghbr, result, image, images_, Bearings_, Rs_, ts_,  patch_size_);
          //AssignPixel(result, i, j, depth, plane, score, nghbr);
          cv::Vec3d normal=cv::Vec3d(UniformRand(-1, 1), UniformRand(-1, 1), UniformRand(-1, 1));
          cv::Vec3d plane = PlanePosFromDepthAndNormal_equirect(Bearings_, j, i, depth, normal);
          //CheckPlaneCandidate(result, i, j, plane, Bearings_, image,images_,Rs_,ts_, patch_size_);
          ComputePlaneScore(i, j, plane, &score, &nghbr, result, image, images_, Bearings_, Rs_, ts_,  patch_size_);
          AssignPixel(result, i, j, depth, plane, score, nghbr);
          
             
        }
      }
    }

  }
  catch (...) {
    std::exception_ptr p = std::current_exception();
    std::cout << (p ? p.__cxa_exception_type()->name() : "null") << std::endl;
  }
}

void ComputeIgnoreMask(DepthmapEstimatorResult *result, std::vector<cv::Mat> &images_, std::vector<cv::Mat> &masks_, int patch_size_, float min_patch_variance_) {
  int hpz = (patch_size_ /2);/// 2;
  //#pragma omp parallel for num_threads(4)
  for (int i = hpz; i < result->depth.rows - hpz; i++) {
    for (int j = hpz; j < result->depth.cols - hpz; j++) {
      bool masked = masks_[0].at<unsigned char>(i, j) == 0;
      bool low_variance = PatchVariance(images_, i, j, patch_size_) < min_patch_variance_;
      bool strong_dist = false; //i > result->depth.rows * 0.8;
      if (masked || low_variance || strong_dist) {
        masks_[0].at<unsigned char>(i, j) = 0;
        AssignPixel(result, i, j, 0.0f, cv::Vec3d(0, 0, 0), 0.0f, 0);
      }
    }
  }
}

float PatchVariance(std::vector<cv::Mat> &images_, int i, int j, int patch_size_) {
  float patch[patch_size_ * patch_size_];
  int hpz = (patch_size_ - 2);/// 2;
  int counter = 0;
  for (int u = -hpz; u <= hpz; u+=2) {
    for (int v = -hpz; v <= hpz; v+=2) {
      patch[counter++] = images_[0].at<unsigned char>(i + u, j + v);
    }
  }
  return Variance(patch, patch_size_ * patch_size_);
}


/* Red Black pattern passes */

void PatchMatchRedPass(DepthmapEstimatorResult *result,bool sample, int patch_size_, std::mt19937 &rng_, std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_) {
  int adjacent[8][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {0, 3}, {3, 0}, {0, -3}, {-3, 0}};
  int hpz = (patch_size_ - 2);
  #pragma omp parallel for //num_threads(16)
  for (int i = hpz; i < result->depth.rows - hpz; i+=1) {
    for (int j = hpz + (i%2); j < result->depth.cols - hpz; j+=2) {
      PatchMatchUpdatePixel(result, i, j, adjacent, sample, rng_, unit_normal_, Bearings_, image, images_, Rs_, ts_, patch_size_);
    }
  }
}

void PatchMatchBlackPass(DepthmapEstimatorResult *result,bool sample, int patch_size_, std::mt19937 &rng_, std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_) {
  int adjacent[8][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {0, 3}, {3, 0}, {0, -3}, {-3, 0}};
  int hpz = (patch_size_ - 2);
  #pragma omp parallel for //num_threads(16)
  for (int i = hpz; i < result->depth.rows - hpz; i+=1) {
    for (int j = hpz + ((i+1)%2); j < result->depth.cols - hpz; j+=2) {
      PatchMatchUpdatePixel(result, i, j, adjacent, sample, rng_, unit_normal_, Bearings_, image, images_, Rs_, ts_, patch_size_);
    }
  }
}


void PatchMatchUpdatePixel(DepthmapEstimatorResult *result,
                                              int i, int j, int adjacent[8][2],
                                              bool sample, std::mt19937 &rng_, std::normal_distribution<float> unit_normal_, cv::Mat &Bearings_, cv::Mat &image, 
                                              std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, std::vector<cv::Vec3d> &ts_, int patch_size_) {
  // Ignore pixels with depth == 0.
  if (result->depth.at<float>(i, j) == 0.0f) {
    return;
  }

  // Check neighbors and their planes for adjacent pixels.
  for (int k = 0; k < 8; ++k) {
    int i_adjacent = i + adjacent[k][0];
    int j_adjacent = j + adjacent[k][1];

    // Do not propagate ignored adjacent pixels.
    if (result->depth.at<float>(i_adjacent, j_adjacent) == 0.0f) {
      continue;
    }

    cv::Vec3d plane = result->plane.at<cv::Vec3d>(i_adjacent, j_adjacent);
    CheckPlaneCandidate(result, i, j, plane, Bearings_, image, images_,Rs_,ts_, patch_size_);
  }

  // Check random planes for current neighbor.
  float depth_range = 0.02;
  float normal_range = 0.5;
  //int current_nghbr = result->nghbr.at<int>(i, j);
  for (int k = 0; k < 6; ++k) {
    float current_depth = result->depth.at<float>(i, j);
    float depth = current_depth * exp(depth_range * unit_normal_(rng_));

    cv::Vec3d current_plane = result->plane.at<cv::Vec3d>(i, j);
    if (current_plane(2) == 0.0) {
      continue;
    }
    
    cv::Vec3d normal(current_plane(0) + normal_range * unit_normal_(rng_),
                     current_plane(1) + normal_range * unit_normal_(rng_),
                     current_plane(2) + normal_range * unit_normal_(rng_));

    //cv::Vec3d normal = normal_tmp / cv::norm(normal_tmp);
    cv::Vec3d plane = PlanePosFromDepthAndNormal_equirect(Bearings_, j, i, depth, normal);
   
    //CheckPlaneImageCandidate(result->depth.cols, result->depth.rows, result, i, j, plane, current_nghbr);
    CheckPlaneCandidate(result, i, j, plane, Bearings_, image, images_, Rs_, ts_, patch_size_);

    depth_range *= 0.3;
    normal_range *= 0.8;
  }

  /*
  if (!sample || images_.size() <= 2) {
    return;
  }

  // Check random other neighbor for current plane.
  int other_nghbr = uni_(rng_);
  while (other_nghbr == current_nghbr) {
    other_nghbr = uni_(rng_);
  }

  cv::Vec3d plane = result->plane.at<cv::Vec3d>(i, j);
  CheckPlaneImageCandidate(result->depth.cols, result->depth.rows,result, i, j, plane, other_nghbr);
  */
}

void CheckPlaneCandidate(DepthmapEstimatorResult *result,int i, int j,const cv::Vec3d &plane, cv::Mat &Bearings_, cv::Mat &image, std::vector<cv::Mat> &images_,std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int patch_size_) {
  float score;
  int nghbr;
  ComputePlaneScore(i, j, plane, &score, &nghbr, result, image, images_, Bearings_, Rs_, ts_, patch_size_);
  if (score > result->score.at<float>(i, j)) {
    float depth = DepthOfPlaneBackprojection_equirect(Bearings_, j, i, plane);
    AssignPixel(result, i, j, depth, plane, score, nghbr);
  }
}


void CheckPlaneImageCandidate(DepthmapEstimatorResult *result, int i, int j, const cv::Vec3d &plane,int nghbr, 
cv::Mat &image, std::vector<cv::Mat> &images_, cv::Mat &Bearings_, std::vector<cv::Matx33d> &Rs_,std::vector<cv::Vec3d> &ts_, int patch_size_) {
  float score = ComputePlaneImageScoreUnoptimized_equirect(images_, Bearings_, Rs_, ts_, image.cols, image.rows, i, j, plane, nghbr, result,  patch_size_);
  if (score > result->score.at<float>(i, j)) {
    float depth = DepthOfPlaneBackprojection_equirect(Bearings_, j, i, plane);
    AssignPixel(result, i, j, depth, plane, score, nghbr);
  }
}

void AssignPixel(DepthmapEstimatorResult *result, int i,
                                    int j, const float depth,
                                    const cv::Vec3d &plane, const float score,
                                    const int nghbr, int patchsize) {
  result->depth.at<float>(i, j) = depth;
  result->plane.at<cv::Vec3d>(i, j) = plane;
  result->score.at<float>(i, j) = score;
  result->nghbr.at<int>(i, j) = nghbr;
  //if (patchsize > 0) {
  //    result->patchs.at<int>(i,j) = patchsize;
  //}

}

void ComputePlaneScore(int i, int j, const cv::Vec3d &plane,float *score, int *nghbr, DepthmapEstimatorResult *result, 
cv::Mat &image, std::vector<cv::Mat> &images_, cv::Mat &Bearings_, std::vector<cv::Matx33d> &Rs_, 
  std::vector<cv::Vec3d> &ts_, int patch_size_) {
  *score = -1.0f;
  *nghbr = 0;
  //std::vector<float> scores_;
  for (int other = 1; other < images_.size(); ++other) {
    float image_score = ComputePlaneImageScoreUnoptimized_equirect(images_, Bearings_, Rs_, ts_, image.cols, image.rows, i, j, plane, other, result, patch_size_);
    //scores_.push_back(image_score);
    if (image_score > *score) {
      *score = image_score;
      *nghbr = other;
    }
  }

  //std::sort(scores_.begin(), scores_.end());
  //*score = scores_[scores_.size()/2];
  //*nghbr=1; //dummy
  //float mean_score = images_score / (images_.size()-1);
  //if (mean_score > *score) {
  //    *score = mean_score;
  //    //*nghbr = other;
  //  }
}

float ComputePlaneImageScoreUnoptimized_equirect(std::vector<cv::Mat> &images_, cv::Mat &Bearings_, 
std::vector<cv::Matx33d> &Rs_, std::vector<cv::Vec3d> &ts_, int cols, int rows, int i, int j, 
const cv::Vec3d &plane, int other, DepthmapEstimatorResult *result, int patch_size_) {
  int hpz = patch_size_ - 2;/// 2;
  float im1_center = images_[0].at<unsigned char>(i, j);
  cv::Matx33d rot_cw = Rs_[other];
  cv::Vec3d trans_cw = ts_[other];
  NCCEstimator ncc;

  for (int dy = -hpz; dy <= hpz; dy+=2) {
    for (int dx = -hpz; dx <= hpz; dx+=2) {
      if (i+dy < 0 || j+dx < 0 || i+dy >=rows || j+dx>=cols) {
        continue;
      }
      float im1 = images_[0].at<unsigned char>(i + dy, j + dx);
      cv::Vec3d pos_w = PosPatchPartOnPlane(Rs_[0], ts_[0], Bearings_, i + dy, j + dx, plane);

      float x2, y2;
      bool too_near = ReprojectEquirect(cols, rows, rot_cw, trans_cw, pos_w, x2, y2);

      if (!too_near) {
        float im2 = LinearInterpolation<unsigned char>(images_[other], y2, x2);
        float weight = BilateralWeight(im1 - im1_center, dx, dy);
        ncc.Push(im1, im2, weight);
      }
    }
  }
  return ncc.Get();
}


float BilateralWeight(float dcolor, float dx, float dy) {
  const float dcolor_sigma = 50.0f;
  const float dx_sigma = 5.0f;
  const float dcolor_factor = 1.0f / (2 * dcolor_sigma * dcolor_sigma);
  const float dx_factor = 1.0f / (2 * dx_sigma * dx_sigma);
  return exp(-dcolor * dcolor * dcolor_factor -
             (dx * dx + dy * dy) * dx_factor);
}

void PostProcess(cv::Mat &Bearings, DepthmapEstimatorResult *result, cv::Matx33d &R, cv::Vec3d &t, const float min_depth, const float max_depth, const float min_score) {
  cv::Mat depth_filtered;
  cv::Mat depth_filtered2;
  cv::medianBlur(result->depth, depth_filtered, 3);
  cv::medianBlur(depth_filtered, depth_filtered2, 3);

  //#pragma omp parallel for
  for (int i = 0; i < result->depth.rows; ++i) {
    for (int j = 0; j < result->depth.cols; ++j) {
      float d = result->depth.at<float>(i, j);
      float m = depth_filtered2.at<float>(i, j);
      float sc = result->score.at<float>(i,j);
      cv::Vec3d plane = result->plane.at<cv::Vec3d>(i,j);
      cv::Vec3d bearing = Bearings.at<cv::Vec3d>(i,j);
      float angl_thr_low = 80;
      float angl_thr_high = 100;
      double scalar_prod = plane.dot(bearing);
      double angle = acos( scalar_prod / cv::norm(plane) ) * 180/M_PI;
      /* ||
        (angle > angl_thr_low && angle < angl_thr_high) || (angle > angl_thr_low-180 && angle < angl_thr_high-180)*/
      if (
        d == 0.0 || fabs(d - m) / d > 0.05 || d < min_depth || d > max_depth || sc <= min_score) {
        result->depth.at<float>(i, j) = 0;
        result->score.at<float>(i, j) = -1;
      }
      else {
        result->depth.at<float>(i, j) = depth_filtered2.at<float>(i, j);
      }
    }
  }
  
  cv::Matx33d R_tmp(
    R(0,0), R(0,1), R(0,2),
    R(1,0), R(1,1), R(1,2),
    R(2,0), R(2,1), R(2,2)); 

  result->R = R_tmp;
  result->t = t;
  
}

DepthmapCleaner::DepthmapCleaner()
    : same_depth_threshold_(0.01), min_consistent_views_(2), nDebugShifts_(0), nDepths_(1) {}

void DepthmapCleaner::ComputeBearingTable() {

  for (int i = 0; i< Bearings_.rows; i++) {
    for (int j = 0; j < Bearings_.cols; j++) {
      //cv::Vec3d bearing =  NormBearing_equirect(BearingFromImgPos_equirect(Bearings_.cols, Bearings_.rows, j, i));
      cv::Vec3d bearing =  BearingFromImgPos_equirect(Bearings_.cols, Bearings_.rows, j, i);
      Bearings_.at<cv::Vec3d>(i, j) = bearing / cv::norm(bearing);
    }
  }

}

void DepthmapCleaner::SetSameDepthThreshold(float t) {
  same_depth_threshold_ = t;
}

int DepthmapCleaner::PopHeadUnlessOne() {

  int nViews = depth_estimate_results_.size();
  if (nViews > 1) {
    /*
    Rs_.erase(Rs_.begin());
    ts_.erase(ts_.begin());
    depths_.erase(depths_.begin());
    */
   depth_estimate_results_.erase(depth_estimate_results_.begin());
  }
  return depth_estimate_results_.size();
}

int DepthmapCleaner::DebugShiftHead() {

  
  if (nDebugShifts_ < depth_estimate_results_.size()) {
    /*
    cv::Matx33d R0(Rs_[0]);
    Rs_.erase(Rs_.begin());
    Rs_.emplace_back(R0);

    cv::Vec3d t0(ts_[0]);
    ts_.erase(ts_.begin());
    ts_.emplace_back(t0);
    
    cv::Mat depth0 = depths_[0].clone();
    depths_.erase(depths_.begin());
    depths_.emplace_back(depth0);
    */
    std::swap(depth_estimate_results_[0], depth_estimate_results_[depth_estimate_results_.size()-1]);
    nDebugShifts_ += 1;
  }
  return nDebugShifts_;
}

void DepthmapCleaner::SetMinConsistentViews(int n) {
  min_consistent_views_ = n;
}

void DepthmapCleaner::SetDepthQueueSize(int n) {
  nDepths_ = n;
}

bool DepthmapCleaner::IsReadyForCompute() {
  return depth_estimate_results_.size() >= nDepths_ && nDebugShifts_ < depth_estimate_results_.size();
}

void DepthmapCleaner::AddView(DepthmapEstimatorResult* result) { //cv::Matx33d &R,cv::Vec3d &t, cv::Mat &depth) {

  DepthmapEstimatorResult* result_tmp = new DepthmapEstimatorResult;
  result_tmp->depth = result->depth.clone();
  result_tmp->score = result->score.clone();
  result_tmp->plane = result->plane.clone();
  result_tmp->R = cv::Matx33d(result->R);
  result_tmp->t = cv::Vec3d(result->t);
  depth_estimate_results_.emplace_back(result_tmp);
  //Rs_.emplace_back(R);
  //ts_.emplace_back(t);
  //depths_.emplace_back(depth.clone());
}

void DepthmapCleaner::reset() {

  //Rs_.clear();
  //ts_.clear();
  //depths_.clear();
  depth_estimate_results_.clear();
  nDebugShifts_ = 0;
}

void DepthmapCleaner::swapCenter() {

  int center_id = depth_estimate_results_.size()/2;
  std::swap(depth_estimate_results_[0], depth_estimate_results_[center_id]);

  //std::swap(Rs_[0],Rs_[center_id]);
  //std::swap(ts_[0],ts_[center_id]);
  //std::swap(depths_[0],depths_[center_id]);
}

void DepthmapCleaner::AssignMatrices(DepthmapCleanerResult *result) {
  result->cleaned_depth = cv::Mat(depth_estimate_results_[0]->depth.rows, depth_estimate_results_[0]->depth.cols, CV_32F, 0.0f);
  //result->cleaned_plane = cv::Mat(depth_estimate_results_[0]->depth.rows, depth_estimate_results_[0]->depth.cols, CV_64FC3, 0.0f);
  //result->cleaned_score = cv::Mat(depth_estimate_results_[0]->depth.rows, depth_estimate_results_[0]->depth.cols, CV_32F, -1.0f);
  result->R = cv::Matx33d(depth_estimate_results_[0]->R);
  result->t = cv::Vec3d(depth_estimate_results_[0]->t);

}

void DepthmapCleaner::Clean(DepthmapCleanerResult* result) {

  if (IsReadyForCompute()) {

    swapCenter();
    AssignMatrices(result);
    

    int cols = depth_estimate_results_[0]->depth.cols;
    int rows = depth_estimate_results_[0]->depth.rows;

    Bearings_ = cv::Mat(rows, cols, CV_64FC3, 0.0f); //cv::Mat(depths_[0].rows, depths_[0].cols, CV_64FC3, 0.0f);
    ComputeBearingTable();
  
    //clean_depth = cv::Mat(rows, cols, CV_32F, 0.0f);
    
    //#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        float depth = depth_estimate_results_[0]->depth.at<float>(i, j);
        if (depth < 1e-6) {
          continue;
        }
        cv::Vec3d point_w = Backproject_equirect(Bearings_, j, i, depth, depth_estimate_results_[0]->R, depth_estimate_results_[0]->t);
        int consistent_views = 1;
        for (int other = 1; other < depth_estimate_results_.size(); ++other) {
          float x;
          float y;
          bool too_near = ReprojectEquirect(cols, rows, depth_estimate_results_[other]->R, depth_estimate_results_[other]->t, point_w, x, y);
          
          if (too_near) {
            continue;
          }

          if (!IsInsideImage(depth_estimate_results_[other]->depth, int(y), int(x))) {
            continue;
            }

          
          float depth_of_point = cv::norm((depth_estimate_results_[other]->R * point_w) + depth_estimate_results_[other]->t);
          float depth_at_reprojection =
              LinearInterpolation<float>(depth_estimate_results_[other]->depth, y, x);
          if (fabs(depth_at_reprojection - depth_of_point) <
              depth_of_point * same_depth_threshold_) {
            consistent_views++;
          }
        }
        if (consistent_views >= min_consistent_views_) {
          result->cleaned_depth.at<float>(i, j) = depth_estimate_results_[0]->depth.at<float>(i, j);
          //result->cleaned_plane.at<float>(i, j) = depth_estimate_results_[0]->plane.at<float>(i, j);
          //result->cleaned_score.at<float>(i, j) = depth_estimate_results_[0]->score.at<float>(i, j);
        }
      }
    }
    swapCenter();
  }
  
}

DepthmapPruner::DepthmapPruner() : same_depth_threshold_(0.01), min_views_(3), nDebugShifts_(0), nDepths_(1) {}

void DepthmapPruner::ComputeBearingTable() {

  for (int i = 0; i< Bearings_.rows; i++) {
    for (int j = 0; j < Bearings_.cols; j++) {
      //cv::Vec3d bearing =  NormBearing_equirect(BearingFromImgPos_equirect(Bearings_.cols, Bearings_.rows, j, i));
      cv::Vec3d bearing =  BearingFromImgPos_equirect(Bearings_.cols, Bearings_.rows, j, i);
      Bearings_.at<cv::Vec3d>(i, j) = bearing / cv::norm(bearing);
    }
  }

}

void DepthmapPruner::SetSameDepthThreshold(float t) {
  same_depth_threshold_ = t;
}

void DepthmapPruner::SetMinViews(int min_views) {
  min_views_ = min_views;
}

void DepthmapPruner::SetDepthQueueSize(int n) {
  nDepths_ = n;
}

void DepthmapPruner::AddView(DepthmapCleanerResult* cleaned_result, cv::Mat &Img) {
  /*
  Rs_.emplace_back(R);
  ts_.emplace_back(t);
  depths_.emplace_back(Depth.clone());
  planes_.emplace_back(Plane.clone());
  */
  
  DepthmapCleanerResult* cleaned_r = new DepthmapCleanerResult;
  cleaned_r->cleaned_depth = cleaned_result->cleaned_depth.clone();
  //cleaned_r->cleaned_score = cleaned_result->cleaned_score.clone();
  //cleaned_r->cleaned_plane = cleaned_result->cleaned_plane.clone();
  cleaned_r->R = cv::Matx33d(cleaned_result->R);
  cleaned_r->t = cv::Vec3d(cleaned_result->t);

  cleaned_results_.push_back(cleaned_r);
  images_.emplace_back(Img.clone());

}

bool DepthmapPruner::IsReadyForCompute() {
  return cleaned_results_.size() >= nDepths_ && nDebugShifts_ < cleaned_results_.size();
}

int DepthmapPruner::PopHead() {

  int nResults = cleaned_results_.size();
  if (nResults > 0) {
    DepthmapCleanerResult* cleaned_result = cleaned_results_[0];
    delete cleaned_result;
    cleaned_result = nullptr;
    cleaned_results_.erase(cleaned_results_.begin());
    /*
    Rs_.erase(Rs_.begin());
    ts_.erase(ts_.begin());
    depths_.erase(depths_.begin());
    planes_.erase(planes_.begin());
    */
    images_.erase(images_.begin());
  }
  return cleaned_results_.size();
}

int DepthmapPruner::DebugShiftHead() {

  /*
  if (nDebugShifts_ < depths_.size()) {
    cv::Matx33d R0(Rs_[0]);
    Rs_.erase(Rs_.begin());
    Rs_.emplace_back(R0);

    cv::Vec3d t0(ts_[0]);
    ts_.erase(ts_.begin());
    ts_.emplace_back(t0);
    
    cv::Mat depth0 = depths_[0].clone();
    depths_.erase(depths_.begin());
    depths_.emplace_back(depth0);

    cv::Mat plane0 = planes_[0].clone();
    planes_.erase(planes_.begin());
    planes_.emplace_back(plane0);

    cv::Mat img0 = images_[0].clone();
    images_.erase(images_.begin());
    images_.emplace_back(img0);

    nDebugShifts_ += 1;
  }
  */
  return 0;
}

void DepthmapPruner::reset() {

  /*
  Rs_.clear();
  ts_.clear();
  depths_.clear();
  planes_.clear();
  */
  int nResults = cleaned_results_.size();
  if (nResults > 0) {
    for (int i=0; i < nResults; i++) {
      DepthmapCleanerResult* cleaned_result = cleaned_results_[i];
      delete cleaned_result;
      cleaned_result = nullptr;
    }
    cleaned_results_.clear();
  }

  images_.clear();
  nDebugShifts_ = 0;
  
}

void DepthmapPruner::AssignMatrices(DepthmapPrunerResult* result) {

  int rows = cleaned_results_[0]->cleaned_depth.rows;
  int cols = cleaned_results_[0]->cleaned_depth.cols;
  result->pruned_depth = cv::Mat(rows, cols, CV_32F, 0.0f);
  //result->pruned_plane = cv::Mat(rows, cols, CV_64FC3, 0.0f);
  //result->pruned_score = cv::Mat(rows, cols, CV_32F, -1.0f);
  result->R = cv::Matx33d(cleaned_results_[0]->R);
  result->t = cv::Vec3d(cleaned_results_[0]->t);
}

void DepthmapPruner::swapCenter() {

  int center_id = cleaned_results_.size()/2;
  std::swap(cleaned_results_[0], cleaned_results_[center_id]);
  std::swap(images_[0], images_[center_id]);
  /*
  std::swap(Rs_[0], Rs_[center_id]);
  std::swap(ts_[0], ts_[center_id]);
  std::swap(depths_[0], depths_[center_id]);
  std::swap(planes_[0], planes_[center_id]);
  */  

  //std::swap(Rs_[0],Rs_[center_id]);
  //std::swap(ts_[0],ts_[center_id]);
  //std::swap(depths_[0],depths_[center_id]);
}

/*
void DepthmapPruner::Prune(std::vector<cv::Vec3d> &merged_points,
                           std::vector<cv::Vec3d> &merged_normals,
                           std::vector<cv::Vec3d> &merged_colors,
                           std::vector<cv::Vec2i> &merged_keypts) {
*/
void DepthmapPruner::Prune(DepthmapPrunerResult* result) {
  
  if (images_.size() == 0) {
    return;
  }

  AssignMatrices(result);

  int rows = images_[0].rows;
  int cols = images_[0].cols;
  Bearings_ = cv::Mat(rows, cols, CV_64FC3, 0.0f);
  ComputeBearingTable();

  cv::Mat &rgb = images_[0];
  cv::Mat &depthmap = cleaned_results_[0]->cleaned_depth;
  //cv::Mat &planemap = cleaned_results_[0]->cleaned_plane;
  //cv::Mat &scoremap = cleaned_results_[0]->cleaned_score;
  cv::Matx33d Rinv = cleaned_results_[0]->R.t();
  cv::Matx33d &R = cleaned_results_[0]->R;
  cv::Vec3d &t = cleaned_results_[0]->t;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {

      bool filter_sparse = false;
      float depth = depthmap.at<float>(i, j);
      if (depth  == 0.0f) {
        filter_sparse = true;
      }

      if(i-1>0 && i-1 < rows) {
        filter_sparse = filter_sparse || depthmap.at<float>(i-1, j) == 0.0f;
      }
      if(j-1>0 && j-1 < cols) {
        filter_sparse = filter_sparse || depthmap.at<float>(i, j-1) == 0.0f;
      }
      if(i+1>0 && i+1 < rows) {
        filter_sparse = filter_sparse || depthmap.at<float>(i+1, j) == 0.0f;
      }
      if(j+1>0 && j+1 < cols) {
        filter_sparse = filter_sparse || depthmap.at<float>(i, j+1) == 0.0f;
      }
      if (filter_sparse) {
        result->pruned_depth.at<float>(i, j) = 0.0f;
        continue;
      }

      cv::Vec3d point_w = Backproject_equirect(Bearings_, j, i, depth, R, t);

      /*
      cv::Vec3d normal = cv::normalize(planes_[0].at<cv::Vec3d>(i, j));
      cv::Vec3d point_w = Backproject_equirect(Bearings_, j, i, depth, Rs_[0], ts_[0]);
      float angle_bearing = abs(normal.dot(Bearings_.at<cv::Vec3d>(i,j)));
      cv::Vec3d normal_w = Rs_[0].t() * normal;
      //angle_bearing = acos( angle_bearing) *180/M_PI;
      */

      result->pruned_depth.at<float>(i, j) = depth;
      //result->pruned_plane.at<cv::Vec3d>(i, j) = planemap.at<cv::Vec3d>(i, j);
      //result->pruned_score.at<float>(i, j) =  scoremap.at<float>(i, j);
      
      //result->points3d.push_back(point_w);
      //result->colors.push_back(rgb.at<cv::Vec3b>(i,j));
      //result->keypts.push_back(cv::Vec2i(i,j));

      bool keep = true;
      for (int other = 1; other < cleaned_results_.size(); ++other) {
        float x;
        float y;
        cv::Matx33d &R_other = cleaned_results_[other]->R;
        cv::Vec3d &t_other = cleaned_results_[other]->t;
        cv::Mat &depthmap_other = cleaned_results_[other]->cleaned_depth;
        //cv::Mat &planemap_other = cleaned_results_[other]->cleaned_plane;
        //cv::Mat &scoremap_other = cleaned_results_[other]->cleaned_score;
        ReprojectEquirect(cols, rows, R_other, t_other, point_w, x, y);

        //why not using LinearInterpolation??
        int iu = int(x);
        int iv = int(y);
        cv::Vec3d point_repr = R_other * point_w + t_other;
        float depth_of_point = cv::norm(point_repr);

        //because it is a panorama, its impossible for any point not to be inside image borders
        if (!IsInsideImage(images_[other], iv, iu) || depth_of_point < 1e-5) {
          continue;
        }

        float depth_at_reprojection = depthmap_other.at<float>(iv, iu);//LinearInterpolation<float>(depthmap_other, y,x);//;depthmap_other.at<float>(iv, iu);
        if (depth_at_reprojection == 0.0f || abs(depth_at_reprojection - depth_of_point) > same_depth_threshold_ * depth_of_point) {
          continue;
        }
        /*
        if (depth_at_reprojection >
            (1 - same_depth_threshold_) * depth_of_point) {
          keep = false;
          break;
        }*/
        
        if (depth < depth_at_reprojection) {

          depthmap_other.at<float>(iv, iu) = 0.0f;
          //planemap_other.at<cv::Vec3d>(iv, iu) = cv::Vec3d(0.0,0.0,0.0);
          //scoremap_other.at<float>(iv, iu) = 0.0f;
          

        } else {
          
          result->pruned_depth.at<float>(i, j) = 0.0f;
          //result->pruned_plane.at<cv::Vec3d>(i, j) = cv::Vec3d(0.0,0.0,0.0);
          //result->pruned_score.at<float>(i, j) =  0.0f;
          keep = false;
          break;
        }


        /*
        if (fabs(depth_at_reprojection -  depth_of_point) < same_depth_threshold_ * depth_of_point) {
          if (depth > depth_of_point) {
            keep = false;
            break;
          }
         */ 
          
          //put
          /*
          cv::Vec3d normal_targ = Rs_[other] * normal_w;
          normal_targ = normal_targ / cv::norm(normal_targ);
          float angle_repr = abs((point_repr/ cv::norm(point_repr)).dot(normal_targ));
          //angle_repr = acos(angle_repr/ cv::norm(point_repr))*180/M_PI;
          if (angle_bearing > angle_repr) {
            keep = false;
            break;
          }//end out
          
          
        } else {
          if (depth_at_reprojection > depth_of_point) {
            keep = false;
            break;
          }
        }
        */
      }
      if (keep) {
        //cv::Vec3d R1_normal = Rinv * normal;
        //merged_normals.push_back(R1_normal);

        result->points3d.push_back(point_w);
        result->colors.push_back(rgb.at<cv::Vec3b>(i,j));
        result->keypts.push_back(cv::Vec2i(i,j));
        
        
        /* need colors and kpts!!
        cv::Vec3d color = rgb.at<cv::Vec3b>(i,j);
        merged_colors.push_back(color);
        cv::Vec2i img_pt(i,j);
        merged_keypts.push_back(img_pt);
        */
      }
    }
  }
}

}  // namespace dense
