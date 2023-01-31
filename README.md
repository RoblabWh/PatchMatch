<div align="center">
	<a href="https://www.en.w-hs.de/">
		<img align="left" src="images/w-hs_logo.png" height="70" alt="whs">
	</a>
	<a href="https://rettungsrobotik.de/en/home">
		<img align="right" src="images/drz_logo.png" height="70" alt="drz">
	</a>
</div>
<h1 align="center">PatchMatch</h1>
<h1 align="center">PatchMatch-Stereo-Panorama, a fast dense reconstruction from 360° video images</h1>
<p align="center">
	<strong>Hartmut Surmann</strong>
	·
	<strong>Marc Thurow</strong>
	·
	<strong>Dominik Slomma</strong>
</p>
<h3 align="center">
	<a href="https://arxiv.org/abs/2211.16266">Paper</a>
</h3>
<p align="center">
	<img src="images/demo.gif" alt="" width="75%">
</p>

***Additional material for the paper: "PatchMatch-Stereo-Panorama, a fast dense reconstruction from 360° video images"***

| **Abstract** — This work proposes a new method for real-time dense 3d reconstruction for common 360° action cams, which can be mounted on small scouting UAVs during USAR missions. The proposed method extends a feature based Visual monocular SLAM (OpenVSLAM, based on the popular ORB-SLAM) for robust long-term localization on equirectangular video input by adding an additional densification thread that computes dense correspondences for any given keyframe with respect to a local keyframe-neighboorhood using a PatchMatch-Stereo-approach. While PatchMatch-Stereo-types of algorithms are considered state of the art for large scale Mutli-View-Stereo they had not been adapted so far for real-time dense 3d reconstruction tasks. This work describes a new massivelly parallel variant of the PatchMatch-Stereo-algorithm that differs from current approaches in two ways:
First it supports the equirectangular camera model while other solutions are limited to the pinhole camera model. Second it is optimized for low latency while keeping a high level of completeness and accuracy. To achieve this it operates only on small sequences of keyframes but employs techniques to compensate for the potential loss of accuracy due to the limited number of frames. Results demonstrate that dense 3d reconstruction is possible on a consumer grade laptop with a recent mobile GPU and that it is possible with improved accuracy and completeness over common offline-MVS solutions with comparable quality settings.  | Example of the UAVs with 360° camera ![UAVs](./images/uavs-thumb.jpg)  |
|:-|-:|

**Keywords**: PatchMatch-Stereo, 360°-Panorama, visual monocular SLAM, UAV, Rescue Robotics


## Usage:
### Requirements
* <a href="https://github.com/NVIDIA/nvidia-docker">NVIDIA Container Toolkit</a>
### Build
```
docker build -t pmdvslam --build-arg NUM_THREADS=$(nproc) .
```
### Run
* Start Container
```
xhost +local:
nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -v /HOST/DATA/PATH:/data pmdvslam
```
* Run VSLAM (press "Terminate" to export and exit)
```
./run_video_slam -v /PATH/TO/ORB_VOCAB.dbow2 -c /PATH/TO/CONFIG.yaml -m /PATH/TO/VIDEO [--mask /PATH/TO/MASK] [-p /PATH/TO/DATABASE.msg]
```
* Export .msg to .ply (already included in normal .msg export)
```
python3 ./export_dense_msg_to_ply.py -i /PATH/TO/FILE.msg_dense -o /PATH/TO/OUTPUT.ply
```
### Realtime Demo
* Start Container and mount this repository as docker volume to /data
* Run Demo
```
./run_video_slam -v /data/orb_vocab/orb_vocab.dbow2 -c /data/example/patchmatch/config.yaml -m /data/example/patchmatch/video.mp4 --mask /data/example/patchmatch/mask.png -p /data/example/patchmatch/output.msg --frame-skip 2 --no-sleep
```

## Videos:

* [Dense mapping of a rescue indoor environment (after a fire) with a UAV (+ 360°) camera, Essen: Feb. 2022](https://www.youtube.com/watch?v=joXGfIUy2mc)

* [High quality version of the video submitted to the ssrr 2022: Dense mapping of a tube at DRZ with a 360° camera at a FPV UAV. High quality video submitted to the ssrr2022. OpenVSLAM + gpu PatchMatch for 360° cameras.](https://www.youtube.com/watch?v=ybpNvSNzGto)

* [Deployment of Aerial Robots after a major fire of an industrial hall with hazardous substances, Berlin: Feb. 2021.](https://www.youtube.com/watch?v=mR05-akD4BE&t=180s)

* [360° View of a short flight through a rescue environment (after a fire) with a DJI FPV and a Insta360 One X, Essen Feb. 2022. Quality is reduced from 5.7 K to 2 K (HD).](https://www.youtube.com/watch?v=Pd2__gm0nUE)

* [3D point cloud of a rescue environment (after a fire). Essen Feb. 2022,  OpenVSLAM + cuda implementation of PatchMatch with equirectangular projection (360°).](https://www.youtube.com/watch?v=mhlxL7Xpauc&t=75s)

* [Dense mapping of a rescue env. in real time with a 360° camera on a small first-person view drone. Watch out the localization and mapping in the tubes!](https://www.youtube.com/watch?v=_xzITKJRyek)

* [360° Indoor panorama viewer based on the localization (similar to streetview). Red points a near blue point a far away. Essen Feb. 2022](https://www.youtube.com/watch?v=iFE1kWW_jM4)

## Cite:
@INPROCEEDINGS{10018698,<br>
  author={Surmann, Hartmut and Thurow, Marc and Slomma, Dominik},<br>
  booktitle={2022 IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR)}, <br>
  title={PatchMatch-Stereo-Panorama, a fast dense reconstruction from 360° video images}, <br>
  year={2022},<br>
  volume={},<br>
  number={},<br>
  pages={366-372},<br>
  doi={10.1109/SSRR56537.2022.10018698}}<br>

## Credits:
* <a href="https://arxiv.org/abs/1610.06475">OrbSLAM</a>
* <a href="https://arxiv.org/abs/1910.01122">OpenVSLAM</a>
