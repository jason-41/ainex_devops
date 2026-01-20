
# 标准消息头
std_msgs/Header header

# 标记ID
int32 marker_id

# 标记位姿（相对于相机坐标系）
geometry_msgs/Pose pose

# 到相机的距离（米）
float64 distance

# 水平角度（偏航角，弧度）
float64 horizontal_angle

# 垂直角度（俯仰角，弧度）
float64 vertical_angle

# 法向量夹角（标记平面法线与相机光轴夹角，弧度）
float64 normal_angle

# 检测置信度（0.0-1.0）
float64 confidence