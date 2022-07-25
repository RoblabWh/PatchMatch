# PatchMatch 

***Additional material for the paper: "PatchMatch-Stereo-Panorama, a fast dense reconstruction from 360° video images"***

| **Abstract** — This work deals with the creation of dense 3D reconstructions based on a 360° video in order to give autonomous robots a better possibility to find their way in their environment as well as to provide different forces in operations with a quick overview of the situation. For this purpose, an already existing visual SLAM method, the OpenVSLAM, which is based on the ORB-SLAM, is extended by a PatchMatch-Stereo-Panorama algorithm. Unlike other methods, this work does not convert the panoramic images into perspective methods, but works directly on the equirectangular projection. In order to operate in real time, a parallel propagation scheme was also developed to offload the computation to the GPU. The results were then compared with structure-from-motion and multi-view stereo methods and showed significant differences.  | Example of the UAVs with 360° camera ![UAVs](./images/uavs-thumb.jpg)  |
|:-|-:|

**Keywords**: PatchMatch-Stereo, 360°-Panorama, visual monocular SLAM, UAV, Rescue Robotics

## Videos (at youtube):

Dense mapping of a rescue indoor environment (after a fire) with a UAV (+ 360°) camera, Essen: Feb. 2022
* [![Essen](./images/vid-thumb-3.png)](https://www.youtube.com/watch?v=joXGfIUy2mc "Essen point cloud generation")

High quality version of the video submitted to the ssrr 2022: Dense mapping of a tube at DRZ with a 360° camera at a FPV UAV. High quality video submitted to the ssrr2022. OpenVSLAM + gpu PatchMatch for 360° cameras.
* [![TubeDRZ](./images/vid-thumb-6.png)](https://www.youtube.com/watch?v=ybpNvSNzGto " Tube mapping DRZ")

Deployment of Aerial Robots after a major fire of an industrial hall with hazardous substances, Berlin: Feb. 2021. started at 3:00
* [![3D point cloud DRZ](./images/vid-thumb-5.png)](https://www.youtube.com/watch?v=mR05-akD4BE&t=180s "Point cloud generation of an burned industrial hall")

360° View of a short flight through a rescue environment (after a fire) with a DJI FPV and a Insta360 One X, Essen Feb. 2022. Quality is reduced from 5.7 K to 2 K (HD).
* [![Essen360](./images/vid-thumb-1.png)](https://www.youtube.com/watch?v=v=Pd2__gm0nUE "Essen flight 2 Minutes 360")

3D point cloud of a rescue environment (after a fire). Essen Feb. 2022,  OpenVSLAM + cuda implementation of PatchMatch with equirectangular projection (360°). Interesting at 1:15
* [![Essenpcl](./images/vid-thumb-2.png)](https://www.youtube.com/watch?v=mhlxL7Xpauc&t=75s "Essen dense point cloud")

Dense mapping of a rescue env. in real time with a 360° camera on a small first-person view drone. Watch out the localization and mapping in the tubes!
* [![DRZMapping](./images/vid-thumb-4.png)](https://www.youtube.com/watch?v=_xzITKJRyek "DRZ mapping")

360° Indoor panorama viewer based on the localization (similar to streetview). Red points a near blue point a far away. Essen Feb. 2022
* [![EssenStreetview](./images/vid-thumb-7.png)](https://www.youtube.com/watch?v=iFE1kWW_jM4 "Essen 360 view")

## Cite:
Hartmut Surmann, Marc Thurow, Dominik Slomma: 
**PatchMatch-Stereo-Panorama, a fast dense reconstruction from 360° video images**, 7 / 2022

## Credits: 
* OrbSLAM https://github.com/raulmur/ORB_SLAM2
* OpenVSLAM https://dl.acm.org/doi/10.1145/3530839.3530849
