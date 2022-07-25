# PatchMatch

***Additional material for the paper: "PatchMatch-Stereo-Panorama, a fast dense reconstruction from 360° video images"***

**Abstract** — This work deals with the creation of dense 3D reconstructions based on a 360° video in order to give autonomous
robots a better possibility to find their way in their environment as well as to provide different forces in operations with a quick
overview of the situation. For this purpose, an already existing visual SLAM method, the OpenVSLAM, which is based on the
ORB-SLAM, is extended by a PatchMatch-Stereo-Panorama algorithm. Unlike other methods, this work does not convert
the panoramic images into perspective methods, but works directly on the equirectangular projection. In order to operate
in real time, a parallel propagation scheme was also developed to offload the computation to the GPU. The results were then
compared with structure-from-motion and multi-view stereo methods and showed significant differences.

**Keywords**: PatchMatch-Stereo, 360°-Panorama, visual monocular SLAM, UAV, Rescue Robotics

## Videos (at youtube):

Dense mapping of a rescue indoor environment (after a fire) with a UAV (+ 360°) camera, Essen: Feb. 2022
* [![Essen](./images/vid-thumb-3.png)](https://www.youtube.com/watch?v=joXGfIUy2mc "Essen point cloud generation")

High quality version of the video submitted to the ssrr 2022: Dense mapping of a tube at DRZ with a 360° camera at a FPV UAV. High quality video submitted to the ssrr2022. OpenVSLAM + gpu PatchMatch for 360° cameras.
* [![TubeDRZ](./images/vid-thumb-6.png)](https://www.youtube.com/watch?v=ybpNvSNzGto " Tube mapping DRZ")

Deployment of Aerial Robots after a major fire of an industrial hall with hazardous substances, Berlin: Feb. 2021. started at 3:00
* [![3D point cloud DRZ](./images/vid-thumb-5.png)](https://www.youtube.com/watch?v=mR05-akD4BE&t=180s "Point cloud generation of an burned industrial hall")
