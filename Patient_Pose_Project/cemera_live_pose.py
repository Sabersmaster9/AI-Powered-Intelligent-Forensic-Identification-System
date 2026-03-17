import cv2
import os
import numpy as np
from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

register_all_modules()

# --- 1. 环境配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'weights', 'yolo26n.pt')
POSE_CONFIG = os.path.join(BASE_DIR, 'weights', 'rtmpose-m_config.py')
POSE_CHECKPOINT = os.path.join(BASE_DIR, 'weights', 'rtmpose-m.pth')

# COCO 17 点索引与连线
COCO_17_INDICES = list(range(17))
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

# --- 2. 初始化模型 ---
print("正在初始化检测与姿态模型...")
det_model = YOLO(YOLO_PATH)
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device='cuda:0')


def is_patient_by_color(crop):
    """颜色过滤逻辑 (现场实验时可微调数值)"""
    if crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 蓝色系范围
    lower_blue = np.array([80, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ratio = np.sum(mask > 0) / (crop.shape[0] * crop.shape[1])
    return ratio > 0.08


# --- 3. 开启摄像头 ---
# 通常内置摄像头为 0，外接 USB 摄像头可能是 1 或 2
cap = cv2.VideoCapture(0)

# 设置分辨率 (如果卡顿可以调低，如 640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("摄像头已就绪，按 'Q' 退出程序...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 为了保证实时性，每帧都检测
    det_results = det_model(frame, verbose=False)
    patient_bbox = None

    for r in det_results:
        for box in r.boxes:
            if int(box.cls) == 0:  # 类别为人
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 防止越界裁剪
                crop = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]
                if is_patient_by_color(crop):
                    patient_bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                    break

    if patient_bbox is not None:
        # 推理 17 点
        pose_results = inference_topdown(pose_model, frame, patient_bbox)
        inst = pose_results[0].pred_instances
        kpts = inst.keypoints[0]
        scores = inst.keypoint_scores[0]

        # 绘制主关节
        for idx in COCO_17_INDICES:
            if scores[idx] > 0.3:
                px, py = kpts[idx].astype(int)
                cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)

        # 绘制骨架连线
        for start, end in SKELETON_17:
            if scores[start] > 0.3 and scores[end] > 0.3:
                pt1 = tuple(kpts[start].astype(int))
                pt2 = tuple(kpts[end].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # UI 提示
        bx = patient_bbox[0].astype(int)
        cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
        cv2.putText(frame, "LIVE: Patient Locked", (bx[0], bx[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示画面
    cv2.imshow('MMPose Live Experiment', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()