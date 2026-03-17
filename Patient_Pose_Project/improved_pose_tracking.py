import cv2
import os
import numpy as np
from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer

register_all_modules()

# --- 1. 配置与路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'weights', 'yolo26n.pt')
POSE_CONFIG = os.path.join(BASE_DIR, 'weights', 'rtmpose-m_config.py')
POSE_CHECKPOINT = os.path.join(BASE_DIR, 'weights', 'rtmpose-m.pth')
VIDEO_INPUT = os.path.join(BASE_DIR, 'data', 'test2.mp4')
VIDEO_OUTPUT = os.path.join(BASE_DIR, 'outputs', 'improved_result.mp4')

# 定义 COCO 17 点索引 (在 Wholebody 133 点中的前 17 位通常对应主关节)
# 顺序依次为: 鼻, 眼, 耳, 肩, 肘, 腕, 胯, 膝, 踝
COCO_17_INDICES = list(range(17))

# 定义 17 点的连线关系 (用于手动绘图)
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

# --- 2. 初始化模型 ---
det_model = YOLO(YOLO_PATH)
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device='cuda:0')


def is_patient_by_color(crop):
    """保持原有的颜色过滤逻辑"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ratio = np.sum(mask > 0) / (crop.shape[0] * crop.shape[1]) if crop.size > 0 else 0
    return ratio > 0.08


# --- 3. 视频处理 ---
cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # YOLO 检测病人
    det_results = det_model(frame, verbose=False)
    patient_bbox = None
    for r in det_results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if is_patient_by_color(frame[max(0, y1):y2, max(0, x1):x2]):
                    patient_bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                    break

    if patient_bbox is not None:
        # 推理
        pose_results = inference_topdown(pose_model, frame, patient_bbox)

        # --- 核心改进：只提取 17 个主关节 ---
        # pose_results[0].pred_instances 包含 keypoints (N, 133, 2)
        inst = pose_results[0].pred_instances
        kpts = inst.keypoints[0]  # 取出第一个人的 133 个点
        scores = inst.keypoint_scores[0]  # 置信度

        # 绘制主关节和骨架
        for idx in COCO_17_INDICES:
            px, py = kpts[idx].astype(int)
            score = scores[idx]
            if score > 0.3:  # 只画置信度高的点
                cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)

        for start_idx, end_idx in SKELETON_17:
            if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
                pt1 = tuple(kpts[start_idx].astype(int))
                pt2 = tuple(kpts[end_idx].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # 绘制 UI 标识
        bx = patient_bbox[0].astype(int)
        cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Patient_1", (bx[0], bx[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Improved Pose', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.write(frame)
cv2.destroyAllWindows()