# Calibration Workflow (easy_handeye)

Source: reformatted from `24-03-2026.txt`.

## 0) Start the ROS Docker container

```bash
docker start ros_noetic_handeye
```
Start the prepared ROS Noetic container for calibration.

```bash
docker exec -it ros_noetic_handeye bash
```
Open a shell inside the container (run this in each terminal T1..T10).

## 1) Launch required nodes (one terminal per role)

### T1 — Robot driver

```bash
roslaunch franka_control franka_control.launch robot_ip:=192.168.10.100 load_gripper:=true
```
Starts Franka low-level control and enables the gripper.

### T2 — Camera acquisition

```bash
rosrun usb_cam usb_cam_node \
  _video_device:=/dev/video0 \
  _pixel_format:=yuyv \
  _image_width:=2560 \
  _image_height:=720 \
  _camera_name:=head_camera \
  _camera_info_url:=file:///root/.ros/camera_info/head_camera.yaml \
  _io_method:=mmap
```
Publishes raw USB camera frames and intrinsics.

### T6 — Camera crop/remap

```bash
rosrun nodelet nodelet standalone image_proc/crop_decimate \
  _x_offset:=0 _y_offset:=0 _width:=1280 _height:=720 \
  camera/image_raw:=/usb_cam/image_raw \
  camera/camera_info:=/usb_cam/camera_info \
  camera_out/image_raw:=/left_cam/image_raw \
  camera_out/camera_info:=/left_cam/camera_info \
  _camera_info_url:=file:///root/.ros/camera_info/head_camera.yaml
```
Crops and republishes camera streams to `/left_cam/*` used by ArUco.

### T7 — Camera rectification/distortion correction

```bash
ROS_NAMESPACE=left_cam rosrun image_proc image_proc
```
Publishes rectified images and camera model in the `left_cam` namespace.

### T3 — ArUco tracking

```bash
rosrun aruco_ros single \
  _markerId:=582 \
  _markerSize:=0.15 \
  _dictionary:=7 \
  _image_is_rectified:=true \
  _camera_frame:="head_camera" \
  _ref_frame:="head_camera" \
  _marker_frame:="camera_marker" \
  image:=/left_cam/image_rect \
  camera_info:=/left_cam/camera_info
```
Detects the ArUco marker and publishes its pose (`camera_marker`) in camera frame.

### T4 — ArUco visual check

```bash
rosrun rqt_image_view rqt_image_view
```
Opens image viewer to verify marker borders and axes overlay.

### T5 — Robot motion interface

```bash
roslaunch panda_moveit_config move_group.launch load_gripper:=true
```
Starts MoveIt for moving the robot during sample collection.

## 2) Two ways to position camera/marker in TF

### Option A — easy_handeye calibration (accurate, recommended)

#### T8 — Run calibration GUI/node

```bash
roslaunch easy_handeye calibrate.launch \
  eye_on_hand:=false \
  namespace_prefix:=panda_eye_to_hand \
  robot_base_frame:=panda_link0 \
  robot_effector_frame:=panda_link8 \
  tracking_base_frame:=head_camera \
  tracking_marker_frame:=camera_marker \
  move_group:="panda_arm"
```
Runs eye-to-hand calibration and computes camera-to-robot transform from sampled poses.

#### T9 — Publish calibrated transform

```bash
roslaunch easy_handeye publish.launch \
  eye_on_hand:=false \
  namespace_prefix:=panda_eye_to_hand
```
Publishes the transform estimated by easy\_handeye into TF.

### Option B — Manual camera pose estimate (quick fallback)

#### T10 — Static TF from ruler measurements

```bash
rosrun tf static_transform_publisher 0.45 -0.9 0.45 0 0 -1.57 panda_link0 head_camera 100
```
Publishes an approximate fixed transform from robot base to camera frame.

## 3) Quick validation checklist

- ArUco detection is stable in `rqt_image_view`.
- `camera_marker` TF is updated when moving the marker.
- Robot can move safely via MoveIt (`move_group`).
- Calibrated/static `panda_link0 -> head_camera` TF is visible and coherent.
