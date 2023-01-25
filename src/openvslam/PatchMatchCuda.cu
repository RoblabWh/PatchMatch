#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <math_constants.h>
#include <curand_kernel.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>

__constant__ float R12_[9];
__constant__ float t12_[3];

__constant__ float R13_[9];
__constant__ float t13_[3];

__constant__ float R1_inv_[9];
__constant__ float t1_inv_[3];

__constant__ float dcolor_sigma;
__constant__ float dx_sigma;
__constant__ float dcolor_factor;
__constant__ float dx_factor;


__device__ float curand_between (curandState *cs, const float min, const float max)
{
    return (curand_uniform(cs) * (max-min) + min);
}

__device__ void PlaneFromDepthNormal(const float bearing_x, const float bearing_y, const float bearing_z, const float depth, const float normal_x,
		const float normal_y, const float normal_z, float* plane_x, float* plane_y, float* plane_z) {

	float px = depth * bearing_x;
	float py = depth * bearing_y;
	float pz = depth * bearing_z;
	float nom = 1/(normal_x * px + normal_y * py + normal_z * pz);

	*plane_x = normal_x * nom;
	*plane_y = normal_y * nom;
	*plane_z = normal_z * nom;

}


__device__ float ComputeScorePlaneHypothesis(const int patchhalf, const float plane_x, const float plane_y, const float plane_z, float* d_Bearingmap, cudaTextureObject_t* d_texs, const int i, const int j, const int width, const int height) {

	float point_x;
	float point_y;
	float point_z;

	float subpix_target_x=100;
	float subpix_target_y=100;
	float I_tgt;
	float I_src;

	float sum_x = 0.0f;
	float sum_y = 0.0f;
	float sum_xx = 0.0f;
	float sum_yy = 0.0f;
	float sum_xy = 0.0f;
	float sum_w = 0.0f;

	float bearing_x;
	float bearing_y;
	float bearing_z;
	float denom;
	float scale;

	float point_x_cam2;
	float point_y_cam2;
	float point_z_cam2;

	float depth;

	float latitude;
	float longitude;

	float prob;
	float mean_x=0;
	float mean_y=0;
	float mean_xx=0;
	float mean_yy=0;
	float mean_xy=0;
	float var_x=0;
	float var_y=0;
	float score_;
	float score;

	float x;
	float y;

	score = -1.0f;
	float* R_targ; float* t_targ;
	R_targ = R12_;
	t_targ = t12_;
	float I_center = tex2D<float>(d_texs[0], j+0.5f, i+0.5f);
	float I_diff;
	float w;
    #pragma unroll(2)
	for (int r = 0; r < 2; r++) {

		#pragma unroll(7)
		for (int k1 = -patchhalf; k1 <= patchhalf; k1+=2) {
			#pragma unroll(7)
			for (int k2  = -patchhalf; k2 <= patchhalf; k2+=2) {
				x = j+k2;
				y = i+k1;
				I_src = tex2D<float>(d_texs[0], x+0.5f, y+0.5f);//f[y+patchhalf][x+patchhalf];//tex2D<float>(d_texs[0], x+0.5f, y+0.5f);//f[threadIdx.y*36+threadIdx.x];//tex2D<float>(d_texs[0], x+0.5f, y+0.5f);//f[threadIdx.y*32+threadIdx.x];//tex2D<float>(d_texs[0], x+0.5f, y+0.5f);

				bearing_x = d_Bearingmap[(3*i+3*k1)*width + 3*j + 3*k2];
				bearing_y = d_Bearingmap[(3*i+3*k1)*width + 3*j + 3*k2 + 1];
				bearing_z = d_Bearingmap[(3*i+3*k1)*width + 3*j + 3*k2 + 2];
				//PointFromPlaneBearing(bearing_x,bearing_y, bearing_z, plane_x,plane_y, plane_z, &point_x, &point_y, &point_z);


				scale = plane_x * bearing_x + plane_y * bearing_y + plane_z * bearing_z;
				denom = 1.0f/scale;

				float point_x_c = bearing_x * denom;
				float point_y_c = bearing_y * denom;
				float point_z_c = bearing_z * denom;

				point_x = R1_inv_[0] * point_x_c + R1_inv_[1] * point_y_c + R1_inv_[2] * point_z_c + t1_inv_[0];
				point_y = R1_inv_[3] * point_x_c + R1_inv_[4] * point_y_c + R1_inv_[5] * point_z_c + t1_inv_[1];
				point_z = R1_inv_[6] * point_x_c + R1_inv_[7] * point_y_c + R1_inv_[8] * point_z_c + t1_inv_[2];

				point_x_cam2 = R_targ[0] * point_x + R_targ[1] * point_y + R_targ[2] * point_z + t_targ[0];
				point_y_cam2 = R_targ[3] * point_x + R_targ[4] * point_y + R_targ[5] * point_z + t_targ[1];
				point_z_cam2 = R_targ[6] * point_x + R_targ[7] * point_y + R_targ[8] * point_z + t_targ[2];

				depth = norm3df(point_x_cam2,point_y_cam2,point_z_cam2);
				scale = 1.0f/fmaxf(1e-6, depth);

				bearing_x = point_x_cam2 * scale;
				bearing_y = point_y_cam2 * scale;
				bearing_z = point_z_cam2 * scale;

				latitude = -asinf(bearing_y);
				longitude = atan2f(bearing_x, bearing_z);

				// convert to pixel image coordinated
				subpix_target_x = width * (0.5f + longitude / (2.0f * CUDART_PI_F));
				subpix_target_y = height * (0.5f - latitude / CUDART_PI_F);

				I_tgt = tex2D<float>(d_texs[r+1], subpix_target_x+0.5f, subpix_target_y+0.5f);

				I_diff = I_src - I_center;
				w = exp(- I_diff * I_diff * dcolor_factor - (k2 * k2 + k1 * k1) * dx_factor);
				sum_x = sum_x + I_src * w;
				sum_y = sum_y + I_tgt * w;
				sum_xx = sum_xx + I_src * I_src * w;
				sum_yy = sum_yy + I_tgt * I_tgt * w;
				sum_xy = sum_xy + I_src * I_tgt * w;


				sum_w = sum_w + w;
			}
		}

		prob = 1/sum_w;
		mean_x = sum_x * prob;
		mean_y = sum_y * prob;
		mean_xx = sum_xx * prob;
		mean_yy = sum_yy * prob;
		mean_xy = sum_xy * prob;
		var_x = mean_xx - mean_x * mean_x;
		var_y = mean_yy - mean_y * mean_y;
		score_= (mean_xy - mean_x * mean_y) / sqrtf(var_x * var_y);
		float score2 = (var_x < 0.1 || var_y < 0.1) ? -1.0f : score_;
		score = (abs((score-score2) / score2) < 0.25) ? 0.5*(score+score2):fmaxf(score, score2);

		R_targ = R13_;
		t_targ = t13_;

	}

	return score;

}

// Device code (Kernel, GPU)
__global__ void //__launch_bounds__(1024,1)
ComputeBearingsEquirect_(float* d_Bearingmap, const int width, const int height)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    float x = j;
    float y = i;

    float lon = 2*CUDART_PI_F * (x/width-0.5f);
    float lat = -CUDART_PI_F * (y/height-0.5f);
    float cos_lat = cosf(lat); // __cosf(lat);
    float cos_lon =cosf(lon);
    float sin_lon = sinf(lon);
    float sin_lat = sinf(lat);

    float3 bearing = make_float3(cos_lat*sin_lon, -sin_lat, cos_lat*cos_lon);
    float norm = 1/norm3df(bearing.x, bearing.y, bearing.z);
    bearing.x*=norm;
    bearing.y*=norm;
    bearing.z*=norm;
	d_Bearingmap[3*i*width + 3*j + 0] = bearing.x;//R1_inv_[0] * bearing.x + R1_inv_[1] * bearing.y + R1_inv_[2] * bearing.z + t1_inv_[0];//bearing.x;//R1_inv_[0] * bearing.x + R1_inv_[1] * bearing.y + R1_inv_[2] * bearing.z + t1_inv_[0];
	d_Bearingmap[3*i*width + 3*j + 1] = bearing.y;//R1_inv_[3] * bearing.x + R1_inv_[4] * bearing.y + R1_inv_[5] * bearing.z + t1_inv_[1];//bearing.y;//R1_inv_[3] * bearing.x + R1_inv_[4] * bearing.y + R1_inv_[5] * bearing.z + t1_inv_[1];
	d_Bearingmap[3*i*width + 3*j + 2] = bearing.z;//R1_inv_[6] * bearing.x + R1_inv_[7] * bearing.y + R1_inv_[8] * bearing.z + t1_inv_[2];//bearing.z;//R1_inv_[6] * bearing.x + R1_inv_[7] * bearing.y + R1_inv_[8] * bearing.z + t1_inv_[2];


}

__global__ void ComputeRandomDepthmap_(float* d_Depthmap, const int width, const int height, const float min_depth, const float max_depth, curandState* d_StatesMap)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    //curandState state;
    //curand_init(clock64(), i, j, &state);
    float rand_depth = curand_between(&d_StatesMap[i*width + j], min_depth, max_depth);
    d_Depthmap[i*width + j] = rand_depth;

}

__global__ void ComputeRandomPlanemap_(float* d_Planemap, float* d_Depthmap, float* d_Bearingmap,const int width, const int height, curandState* d_StatesMap)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //curandState state;
	//curand_init(clock64(), i, j, &state);
	float rand_normal_x = curand_between(&d_StatesMap[i*width + j], -1.0f, 1.0f);
	//curand_init(clock64(), i, j, &state);
	float rand_normal_y = curand_between(&d_StatesMap[i*width + j], -1.0f, 1.0f);
	//curand_init(clock64(), i, j, &state);
	float rand_normal_z = curand_between(&d_StatesMap[i*width + j], -1.0f, 1.0f);

	float plane_x;
	float plane_y;
	float plane_z;

	PlaneFromDepthNormal(d_Bearingmap[3*i*width + 3*j + 0], d_Bearingmap[3*i*width + 3*j + 1], d_Bearingmap[3*i*width + 3*j + 2], d_Depthmap[i*width + j],
			rand_normal_x, rand_normal_y, rand_normal_z, &plane_x, &plane_y, &plane_z);
    d_Planemap[3*i*width + 3*j + 0] = plane_x;
    d_Planemap[3*i*width + 3*j + 1] = plane_y;
    d_Planemap[3*i*width + 3*j + 2] = plane_z;

}

__global__ void ComputeRandomSeed_(curandState* d_StatesMap, const int width, const int height)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    curand_init(clock64(), i, j, &d_StatesMap[i*width + j]);

}


__global__ void  __launch_bounds__(1024,1)
ScorePlaneDepth_(const int patchhalf, float* d_Scoremap, float* d_Planemap, float* d_Depthmap, float* d_Bearingmap, const int width, const int height, cudaTextureObject_t* d_texs)
{

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < patchhalf+1 || i > height - patchhalf-1 || j < patchhalf+1 || j > width - patchhalf-1 ) {
			return;
	}


	float score = ComputeScorePlaneHypothesis(patchhalf, d_Planemap[3*i*width + 3*j], d_Planemap[3*i*width + 3*j + 1], d_Planemap[3*i*width + 3*j + 2], d_Bearingmap, d_texs, i, j, width, height);
	d_Scoremap[i*width + j] = score;

}


__global__ void __launch_bounds__(1024,1)
PatchMatchRedBlackPass_(const int patchhalf, float* d_Scoremap, float* d_Depthmap, float* d_Planemap, curandState* d_StatesMap,
		float* d_Bearingmap, cudaTextureObject_t* d_texs, const int width, const int height, const int passing_offset) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	j *= 2;
	j -= ((i+passing_offset)%2);

	if (i < patchhalf+1 || i > height - patchhalf-1 || j < patchhalf+1 || j > width - patchhalf-1 ) {
		return;
	}

	float current_depth;

	int i_n;
	int j_n;

	float score = d_Scoremap[i*width + j];
	float score_hyp;
	float plane_x_n;
	float plane_y_n;
	float plane_z_n;
	float plane_x_hyp = d_Planemap[i*3*width + 3*j];
	float plane_y_hyp = d_Planemap[i*3*width + 3*j + 1];
	float plane_z_hyp = d_Planemap[i*3*width + 3*j + 2];

	float bearing_x = d_Bearingmap[i*3*width + 3*j];
	float bearing_y = d_Bearingmap[i*3*width + 3*j + 1];
	float bearing_z = d_Bearingmap[i*3*width + 3*j + 2];
	//1. check neighboors planes for pixel at this i,j
	int adjacent[8][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0},{0, 3}, {3, 0}, {0, -3}, {-3, 0}};

	float denom;
	float curr_depth = d_Depthmap[i*width + j];


	#pragma unroll(8)
	for (int a = 0; a < 8; a++) {
		i_n = adjacent[a][0] + i;
		j_n = adjacent[a][1] + j;

		plane_x_n = d_Planemap[i_n*3*width + 3*j_n];
		plane_y_n = d_Planemap[i_n*3*width + 3*j_n + 1];
		plane_z_n = d_Planemap[i_n*3*width + 3*j_n + 2];

		score_hyp = ComputeScorePlaneHypothesis(patchhalf, plane_x_n, plane_y_n, plane_z_n, d_Bearingmap, d_texs, i, j, width, height);

		float scale = plane_x_n * bearing_x + plane_y_n * bearing_y + plane_z_n * bearing_z;
		denom = 1/fmaxf(1e-6f, fabsf(scale)); //depth

		//scalar_prod / cv::norm(plane)
		float depth_smooth_f = expf(-powf((curr_depth-denom) /curr_depth,2) * 1250.0f);
		float prod = plane_x_hyp*plane_x_n + plane_y_hyp*plane_y_n + plane_z_hyp*plane_z_n;
		float norm = sqrtf(powf(plane_x_hyp-plane_x_n,2) + powf(plane_y_hyp-plane_y_n,2) + powf(plane_z_hyp-plane_z_n,2));

		float plane_smooth_f = expf(-powf(acosf(prod/norm),2) * 8.0f);
		score_hyp *= (1.0f-depth_smooth_f) * (1.0f-plane_smooth_f);
		if (score_hyp > score) {
			plane_x_hyp = plane_x_n;
			plane_y_hyp = plane_y_n;
			plane_z_hyp = plane_z_n;
			score = score_hyp;
		}
	}

	denom = bearing_x * plane_x_hyp + bearing_y * plane_y_hyp + bearing_z * plane_z_hyp;
	current_depth = 1/fmaxf(1e-6f, fabsf(denom));
	//2. random guessing phase

	// Check random planes for current neighbor.
	float px;
	float py;
	float pz;
	float depth_range = 0.02;
	float normal_range = 0.5;
	float current_plane_x = plane_x_hyp;
	float current_plane_y = plane_y_hyp;
	float current_plane_z = plane_z_hyp;

	float rand_a;
	float rand_b;
	float rand_c;
	float depth_hyp;

	float normal_hyp_x;
	float normal_hyp_y;
	float normal_hyp_z;

	#pragma unroll(6)
	for (int t = 0; t < 6; ++t) {
		rand_a = curand_normal(&d_StatesMap[i*width + j]);
		rand_b = curand_normal(&d_StatesMap[i*width + j]);
		rand_c = curand_normal(&d_StatesMap[i*width + j]);
		depth_hyp = current_depth * expf(depth_range * rand_a); //__expf(depth_range * rand_a);

		normal_hyp_x = current_plane_x + normal_range * rand_a;
		normal_hyp_y = current_plane_y + normal_range * rand_b;
		normal_hyp_z = current_plane_z + normal_range * rand_c;

		//PlaneFromDepthNormal(bearing_x, bearing_y, bearing_z, depth_hyp, normal_hyp_x, normal_hyp_y, normal_hyp_z, &plane_x_hyp, &plane_y_hyp, &plane_z_hyp);
		px = depth_hyp * bearing_x;
		py = depth_hyp * bearing_y;
		pz = depth_hyp * bearing_z;
		denom = normal_hyp_x * px + normal_hyp_y * py + normal_hyp_z * pz;

		plane_x_hyp = normal_hyp_x / denom;
		plane_y_hyp = normal_hyp_y / denom;
		plane_z_hyp = normal_hyp_z / denom;

		score_hyp = ComputeScorePlaneHypothesis(patchhalf, plane_x_hyp, plane_y_hyp, plane_z_hyp, d_Bearingmap, d_texs, i, j, width, height);
		if (score_hyp > score) {
			current_plane_x = plane_x_hyp;
			current_plane_y = plane_y_hyp;
			current_plane_z = plane_z_hyp;
			score = score_hyp;
			current_depth = depth_hyp;
		}

		depth_range *= 0.3;
		normal_range *= 0.8;
	}

	d_Scoremap[i*width + j] = score;
	d_Planemap[i*3*width + 3*j] = current_plane_x;
	d_Planemap[i*3*width+ 3*j + 1] = current_plane_y;
	d_Planemap[i*3*width + 3*j + 2] = current_plane_z;
	//current_depth = DepthFromPlaneBearing(bearing_x, bearing_y, bearing_z, current_plane_x, current_plane_y, current_plane_z);
	denom = bearing_x * current_plane_x + bearing_y * current_plane_y + bearing_z * current_plane_z;
	current_depth = 1/fmaxf(1e-6f, fabsf(denom));
	d_Depthmap[i*width + j] = current_depth;


}

__global__ void testTextImgs_(cudaTextureObject_t* d_texs) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	float f1 = tex2D<float>(d_texs[0],j,i);
	float f2 = tex2D<float>(d_texs[1],j,i);


}

extern "C" void ComputeBearingsEquirect(float* d_Bearingmap, const int width, const int height) {

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Der Kernelaufruf erfolgt
	// Festlegung der Threads pro Block
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	// Der Kernel wird gestartet
	ComputeBearingsEquirect_<<<numBlocks, threadsPerBlock>>>(d_Bearingmap, width, height);

	// Wait for GPU to finish before accessing on host
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("DEBUG: Time spent bearing map: %.8f ms\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << cudaGetErrorString(err) << std::endl;
	}

}

extern "C" void ComputeRandomPlanemapDepthmap(float* d_Depthmap, float* d_Planemap,float*  d_Bearingmap, int width, int height, float min_depth, float max_depth, curandState* d_StatesMap) {

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Der Kernelaufruf erfolgt
	// Festlegung der Threads pro Block
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	// Der Kernel wird gestartet
	ComputeRandomDepthmap_<<<numBlocks, threadsPerBlock>>>(d_Depthmap, width, height, min_depth, max_depth, d_StatesMap);
	ComputeRandomPlanemap_<<<numBlocks, threadsPerBlock>>>(d_Planemap, d_Depthmap, d_Bearingmap, width, height, d_StatesMap);

	// Wait for GPU to finish before accessing on host
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("DEBUG: Time spent random depth and plane maps: %.8f ms\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << cudaGetErrorString(err) << std::endl;
	}

}

extern "C" void ComputeRandomPlanemap(float* d_Planemap, float* d_Depthmap, float* d_Bearingmap, int width, int height, curandState* d_StatesMap) {

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Der Kernelaufruf erfolgt
	// Festlegung der Threads pro Block
	dim3 threadsPerBlock(8,8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	// Der Kernel wird gestartet
	ComputeRandomPlanemap_<<<numBlocks, threadsPerBlock>>>(d_Planemap, d_Depthmap, d_Bearingmap, width, height, d_StatesMap);

	// Wait for GPU to finish before accessing on host
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("DEBUG: Time spent random planemap: %.8f ms\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << cudaGetErrorString(err) << std::endl;
	}

}

extern "C" void ScorePlaneDepth(const int patchhalf, float* d_Scoremap, float* d_Depthmap, float* d_Planemap, float* d_Bearingmap, cudaTextureObject_t* d_texs, int width, int height) {

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Der Kernelaufruf erfolgt
	// Festlegung der Threads pro Block
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	//dim3 threadsPerBlock(16,24);
	//dim3 numBlocks(width/threadsPerBlock.x, (height+23)/threadsPerBlock.y);

	// Der Kernel wird gestartet
	//std::cout << std::to_string(Img_targ[(width-1)*(height-1)]) << std::endl;
	ScorePlaneDepth_<<<numBlocks, threadsPerBlock, 36*36*sizeof(float)>>>(patchhalf, d_Scoremap, d_Planemap, d_Depthmap, d_Bearingmap, width, height, d_texs);

	// Wait for GPU to finish before accessing on host
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("DEBUG: Time spent scoring: %.8f ms\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << cudaGetErrorString(err) << std::endl;
	}
}


extern "C" void PatchMatchRedBlackPass(const int patchhalf, int nIterations, float* d_Scoremap, float* d_Depthmap, float* d_Planemap, float* d_Bearingmap, cudaTextureObject_t* d_texs, int width, int height, curandState* d_StatesMap) {

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Der Kernelaufruf erfolgt
	// Festlegung der Threads pro Block
	dim3 threadsPerBlock(32,32);
	//dim3 numBlocks((width/threadsPerBlock.x)/2, height/threadsPerBlock.y);
	dim3 numBlocks((width/threadsPerBlock.x)/2, height/threadsPerBlock.y);


	// Der Kernel wird gestartet
	int offset_black_pass;
	for(int i = 0; i < nIterations; i++) {
		offset_black_pass = 0;
		PatchMatchRedBlackPass_<<<numBlocks, threadsPerBlock, 36*36*sizeof(float)>>>(patchhalf, d_Scoremap, d_Depthmap, d_Planemap, d_StatesMap, d_Bearingmap, d_texs, width, height, offset_black_pass);
		offset_black_pass = 1;
		cudaThreadSynchronize();
		PatchMatchRedBlackPass_<<<numBlocks, threadsPerBlock, 36*36*sizeof(float)>>>(patchhalf, d_Scoremap, d_Depthmap, d_Planemap, d_StatesMap, d_Bearingmap, d_texs, width, height, offset_black_pass);
		cudaThreadSynchronize();
	}


	// Wait for GPU to finish before accessing on host
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("DEBUG: Time spent propagation: %.8f ms\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << cudaGetErrorString(err) << std::endl;
	}



}

extern "C" void PostProcess() {

}

extern "C" void ComputeRandomSeed(curandState* d_StatesMap, const int width, const int height) {

	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Der Kernelaufruf erfolgt
	// Festlegung der Threads pro Block
	dim3 threadsPerBlock(8,8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	// Der Kernel wird gestartet
	ComputeRandomSeed_<<<numBlocks, threadsPerBlock>>>(d_StatesMap, width, height);

	// Wait for GPU to finish before accessing on host
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("DEBUG: Time spent seed map: %.8f ms\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << cudaGetErrorString(err) << std::endl;
	}
}

extern "C" void RandomInitialization(const int patchhalf, float* d_Scoremap, float* d_Depthmap, float* d_Planemap, float* d_Bearingmap,
		cudaTextureObject_t* d_texs, int width, int height, float min_depth, float max_depth, curandState* d_StatesMap) {

	//ComputeRandomSeed(d_StatesMap, width, height);
	ComputeRandomPlanemapDepthmap(d_Depthmap, d_Planemap, d_Bearingmap, width, height, min_depth, max_depth, d_StatesMap);
	//ComputeRandomPlanemap(d_Planemap, d_Bearingmap, d_Depthmap, width, height, d_StatesMap);
	ScorePlaneDepth(patchhalf, d_Scoremap, d_Depthmap, d_Planemap, d_Bearingmap, d_texs, width, height);
}

extern "C"
void testTextImages(cudaTextureObject_t *d_texs) {
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(640/threadsPerBlock.x, 320/threadsPerBlock.y);
	testTextImgs_<<<numBlocks,threadsPerBlock>>>(d_texs);

	// Wait for GPU to finish before accessing on host
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("Time spent test: %.8f\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "testTextImages " << cudaGetErrorString(err) << std::endl;
	}

}

extern "C"
void InitTextureImages(const int width, const int height, cv::Mat &Img1, cv::Mat &Img2, cv::Mat &Img3, float* d_img1, float* d_img2, float* d_img3, cudaTextureObject_t texs[]) {


	float* h_img1_= (float*)Img1.data;
	float* h_img2_= (float*)Img2.data;
    float* h_img3_= (float*)Img3.data;
	size_t pitch;

	//cudaMallocArray(&cuArray1, &channelDesc, sizeof(float)*width, height);
	cudaMallocPitch(&d_img1, &pitch, sizeof(float)*width, height);
	cudaMallocPitch(&d_img2, &pitch, sizeof(float)*width, height);
    cudaMallocPitch(&d_img3, &pitch, sizeof(float)*width, height);

	//cudaMemcpy2DToArray(cuArray1, 0, 0, Img1.data, Img1.step[0], width*sizeof(float), height, cudaMemcpyHostToDevice);
	//cudaMallocArray(&cuArray2, &channelDesc, sizeof(float)*width, height);
	//cudaMemcpy2DToArray(cuArray2, 0, 0, Img2.data, Img2.step[0], width*sizeof(float), height, cudaMemcpyHostToDevice);

	cudaMemcpy2D(d_img1, pitch, h_img1_, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_img2, pitch, h_img2_, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_img3, pitch, h_img3_, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice);

	//create image arrays global memor
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	struct cudaResourceDesc resDesc1;
	memset(&resDesc1, 0, sizeof(cudaResourceDesc));
	resDesc1.resType         = cudaResourceTypePitch2D;
	resDesc1.res.pitch2D.devPtr   = d_img1;
	resDesc1.res.pitch2D.desc     = channelDesc;
	resDesc1.res.pitch2D.width    = width;
	resDesc1.res.pitch2D.height   = height;
	resDesc1.res.pitch2D.pitchInBytes = pitch;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc1;
	memset(&texDesc1, 0, sizeof(cudaTextureDesc));
	texDesc1.addressMode[0]   = cudaAddressModeWrap;
	texDesc1.addressMode[1]   = cudaAddressModeWrap;
	texDesc1.filterMode       = cudaFilterModePoint;
	texDesc1.readMode         = cudaReadModeElementType;
	texDesc1.normalizedCoords = false;

	cudaCreateTextureObject(&texs[0], &resDesc1, &texDesc1, NULL);


	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float>();

	struct cudaResourceDesc resDesc2;
	memset(&resDesc2, 0, sizeof(cudaResourceDesc));
	resDesc2.resType         = cudaResourceTypePitch2D;
	resDesc2.res.pitch2D.devPtr   = d_img2;
	resDesc2.res.pitch2D.desc     = channelDesc2;
	resDesc2.res.pitch2D.width    = width;
	resDesc2.res.pitch2D.height   = height;
	resDesc2.res.pitch2D.pitchInBytes = pitch;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc2;
	memset(&texDesc2, 0, sizeof(cudaTextureDesc));
	texDesc2.addressMode[0]   = cudaAddressModeWrap;
	texDesc2.addressMode[1]   = cudaAddressModeWrap;
	texDesc2.filterMode       = cudaFilterModeLinear;
	texDesc2.readMode         = cudaReadModeElementType;//cudaReadModeElementType;
	texDesc2.normalizedCoords = false;

	cudaCreateTextureObject(&texs[1], &resDesc2, &texDesc2, NULL);




    cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc<float>();

	struct cudaResourceDesc resDesc3;
	memset(&resDesc3, 0, sizeof(cudaResourceDesc));
	resDesc3.resType         = cudaResourceTypePitch2D;
	resDesc3.res.pitch2D.devPtr   = d_img3;
	resDesc3.res.pitch2D.desc     = channelDesc3;
	resDesc3.res.pitch2D.width    = width;
	resDesc3.res.pitch2D.height   = height;
	resDesc3.res.pitch2D.pitchInBytes = pitch;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc3;
	memset(&texDesc3, 0, sizeof(cudaTextureDesc));
	texDesc3.addressMode[0]   = cudaAddressModeWrap;
	texDesc3.addressMode[1]   = cudaAddressModeWrap;
	texDesc3.filterMode       = cudaFilterModeLinear;
	texDesc3.readMode         = cudaReadModeElementType;//cudaReadModeElementType;
	texDesc3.normalizedCoords = false;

	cudaCreateTextureObject(&texs[2], &resDesc3, &texDesc3, NULL);


	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "InitTextureImages " << cudaGetErrorString(err) << std::endl;
	}
}


extern "C"
void InitTextureBearingTable(const int width, const int height, float* d_Bearingmap, float* d_Bearingmap_pitch, cudaTextureObject_t* bearing_tex) {

	size_t pitch;

	cudaMallocPitch(&d_Bearingmap_pitch, &pitch, 4*sizeof(float)*width, height);

	//create 3d array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	//cudaExtent bearingSize = make_cudaExtent(width, height, 3);

	//cudaMalloc3DArray(&d_beaArray, &channelDesc, bearingSize);
	float* h_Bearingmap_3 = (float*)malloc(3*sizeof(float)*width*height);
	cudaMemcpy(h_Bearingmap_3, d_Bearingmap, 3*sizeof(float)*width*height, cudaMemcpyDeviceToHost);
	float* h_Bearingmap_4 = (float*)malloc(4*sizeof(float)*width*height);
	int j = 0;
	for (int i = 0; i < 4*width*height; i++) {
		if (i%4==0 && i>0) {
			h_Bearingmap_4[i] = 0;
		} else {
			h_Bearingmap_4[i] = h_Bearingmap_3[j];
			j++;
		}
		//std::cout << std::to_string(h_Bearingmap_4[i]) << std::endl;
	}

	cudaMemcpy2D(d_Bearingmap_pitch, pitch, h_Bearingmap_4, 4*sizeof(float)*width, 4*sizeof(float)*width, height, cudaMemcpyHostToDevice);

	struct cudaResourceDesc resDesc1;
	memset(&resDesc1, 0, sizeof(cudaResourceDesc));
	resDesc1.resType         = cudaResourceTypePitch2D;
	resDesc1.res.pitch2D.devPtr   = d_Bearingmap_pitch;
	resDesc1.res.pitch2D.desc     = channelDesc;
	resDesc1.res.pitch2D.width    = width;
	resDesc1.res.pitch2D.height   = height;
	resDesc1.res.pitch2D.pitchInBytes = pitch;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc1;
	memset(&texDesc1, 0, sizeof(cudaTextureDesc));
	texDesc1.addressMode[0]   = cudaAddressModeClamp;
	texDesc1.addressMode[1]   = cudaAddressModeClamp;
	texDesc1.filterMode       = cudaFilterModePoint;
	texDesc1.readMode         = cudaReadModeElementType;
	texDesc1.normalizedCoords = false;

	cudaCreateTextureObject(&bearing_tex[0], &resDesc1, &texDesc1, NULL);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "InitTextureBearingTable " << cudaGetErrorString(err) << std::endl;
	}

	free(h_Bearingmap_3);
	free(h_Bearingmap_4);
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    printf("\n");
    return;
}

extern "C"
void InitConstantMem(float* R12, float* t12, float* R13, float* t13, float* R1_inv, float* t1_inv) {

	cudaMemcpyToSymbol(R12_, R12, 9*sizeof(float), 0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(t12_, t12, 3*sizeof(float), 0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(R13_, R13, 9*sizeof(float), 0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(t13_, t13, 3*sizeof(float), 0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(R1_inv_, R1_inv, 9*sizeof(float), 0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(t1_inv_, t1_inv, 3*sizeof(float), 0,cudaMemcpyHostToDevice);

	float dcolor_sigma_ = 50.0f;
	float dx_sigma_ = 5.0f;
	float dcolor_factor_ = 1.0f / (2 * dcolor_sigma_ * dcolor_sigma_);
	float dx_factor_ = 1.0f / (2 * dx_sigma_ * dx_sigma_);
	cudaMemcpyToSymbol(dcolor_sigma, &dcolor_sigma_, sizeof(float), 0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dx_sigma, &dx_sigma_, sizeof(float), 0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dcolor_factor, &dcolor_factor_, sizeof(float), 0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dx_factor, &dx_factor_, sizeof(float), 0,cudaMemcpyHostToDevice);

	// Fehlerbehandlung
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "InitConstantMemt " << cudaGetErrorString(err) << std::endl;
	}
}
