## environment setup
```bash
python3 -m venv my_venv --system-site-packages --symlinks
source my_venv/bin/activate
pip install mediapipe

source /opt/ros/jazzy/setup.bash 
source install/setup.bash 
```
## Deactivate the Environment
```
deactivate
```

## Compile
```
colcon build --symlink-install
. install/setup.bash
```

## How to Run
in the workspace dir run in different cmd windows:
```bash
ros2 run face_search_and_wave face_and_wave 
ros2 run ainex_vision face_detection_node
```

## Testing & Developing
### 1. Finding the range or positions of the robot joints
in the file "joint_visualization_node.py" Line 28
```python
joint_names = ['r_sho_pitch', 'r_sho_roll', 'r_el_pitch','r_el_yaw'] #change to your desired joint names if needed
```
then in the workspace dir run in different cmd windows:
```bash
ros2 run ainex_motion joint_visualization
ros2 run plotjuggler plotjuggler
```
### 2. Lock & Unlock
```bash
ros2 service call /Lock_All_Joints std_srvs/srv/Empty {}
ros2 service call /Unlock_All_Joints std_srvs/srv/Empty {}
```

## Waving posistions
### 1. Standby positions(hand down)
r_sho_roll: -0.017   
r_sho_pitch: 0.18  
r_el_pitch: -0.084  
r_el_yaw: 1.257

### 2. Standby positions(hand up)
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0   
r_el_yaw: 1.55  

### 3. Far Right positions
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0   
r_el_yaw: 0.95  

### 4. Far Left positions
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0   
r_el_yaw: 2.0 

### 5. Front positions(optional)
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: -0,75   
r_el_yaw: 1.55  

### 6. Back positions(optional)
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0.75    
r_el_yaw: 1.55  
