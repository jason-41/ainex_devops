## environment setup
```bash
python3 -m venv my_venv --system-site-packages --symlinks
source my_venv/bin/activate
pip install mediapipe

# source hrs_env/bin/activate ##just different env name
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

## Testing the joint coordinates ranges:
in the file "joint_visualization_node.py" Line 28
```python
joint_names = ['r_sho_pitch', 'r_sho_roll', 'r_el_pitch','r_el_yaw'] #change to your desired joint names if needed
```

## Waving posistions:
### 1. Standby positions(hands up)
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0   
r_el_yaw: 1.55  

### 2. Far Right positions
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0   
r_el_yaw: 0.95  

### 3. Far Left positions
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0   
r_el_yaw: 1.55  

### 4. Front positions(optional)
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: -0,75   
r_el_yaw: 1.55  

### 4. Back positions(optional)
r_sho_roll: 1.515   
r_sho_pitch: 1.595  
r_el_pitch: 0.75    
r_el_yaw: 1.55  
