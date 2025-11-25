# FaceBoundingBox.msg
# 单个人脸的 bounding box 数据（相对于图像坐标系）

float32 x_min     # 左上角 x 坐标
float32 y_min     # 左上角 y 坐标
float32 width     # bounding box 宽度
float32 height    # bounding box 高度

float32 score     # 检测置信度（0~1）
