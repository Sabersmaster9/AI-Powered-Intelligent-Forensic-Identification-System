import cv2
import os
import numpy as np
from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

register_all_modules()

# --- 1. 配置与路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'weights', 'yolo26n.pt')
POSE_CONFIG = os.path.join(BASE_DIR, 'weights', 'rtmpose-m_config.py')
POSE_CHECKPOINT = os.path.join(BASE_DIR, 'weights', 'rtmpose-m.pth')
VIDEO_INPUT = os.path.join(BASE_DIR, 'data', 'test1.mp4')
VIDEO_OUTPUT = os.path.join(BASE_DIR, 'outputs', 'improved_result.mp4')

COCO_17_INDICES = list(range(17))
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

# --- 2. 初始化模型 ---
det_model = YOLO(YOLO_PATH)
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device='cuda:0')


def check_color_type(crop):
    """
    返回识别结果: 0-未知, 1-病人(蓝), 2-医生(白)
    """
    if crop.size == 0: return 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 病人：深蓝色区域
    lower_blue = np.array([100, 50, 50])  # 略微收紧了范围，减少杂色干扰
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = np.sum(mask_blue > 0) / (crop.shape[0] * crop.shape[1])

    # 医生：白色区域 (HSV 中饱和度低、亮度高即为白)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.sum(mask_white > 0) / (crop.shape[0] * crop.shape[1])

    if blue_ratio > 0.08: return 1
    if white_ratio > 0.25: return 2  # 医生白大褂面积通常较大
    return 0


# --- 3. 视频处理 ---
cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    det_results = det_model(frame, verbose=False)

    candidates = []  # 存放所有检测到的人及其属性

    for r in det_results:
        for box in r.boxes:
            if int(box.cls) == 0:  # 类别为 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[max(0, y1):y2, max(0, x1):x2]

                # 计算颜色比例
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # 病人判定：深蓝色 (根据你提供的图片，病人的裤子/衣服偏深蓝/灰蓝)
                lower_blue = np.array([100, 40, 40])
                upper_blue = np.array([140, 255, 255])
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                blue_ratio = np.sum(blue_mask > 0) / roi.size if roi.size > 0 else 0

                # 医生判定：纯白色 (白大褂在 HSV 中饱和度低、亮度高)
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 50, 255])
                white_mask = cv2.inRange(hsv, lower_white, upper_white)
                white_ratio = np.sum(white_mask > 0) / roi.size if roi.size > 0 else 0

                candidates.append({
                    'bbox': [x1, y1, x2, y2],
                    'blue_ratio': blue_ratio,
                    'white_ratio': white_ratio
                })

    # 1. 锁定唯一的病人：蓝色得分最高的那位
    if candidates:
        # 按蓝色比例排序，取最高的一个作为病人
        candidates.sort(key=lambda x: x['blue_ratio'], reverse=True)
        patient = candidates[0]

        # 2. 渲染病人 (仅限蓝色得分超过阈值的，或者直接认定最高者为病人)
        px1, py1, px2, py2 = patient['bbox']
        patient_bbox_input = np.array([[px1, py1, px2, py2]], dtype=np.float32)

        # 执行姿态估计
        pose_results = inference_topdown(pose_model, frame, patient_bbox_input)
        inst = pose_results[0].pred_instances
        kpts = inst.keypoints[0]
        scores = inst.keypoint_scores[0]

        # 画病人骨架
        for idx in COCO_17_INDICES:
            if scores[idx] > 0.3:
                cv2.circle(frame, (int(kpts[idx][0]), int(kpts[idx][1])), 5, (0, 255, 0), -1)
        for start_idx, end_idx in SKELETON_17:
            if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
                pt1 = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                pt2 = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(frame, "Patient", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. 渲染医生：处理剩下的 candidate
        for doc in candidates[1:]:
            dx1, dy1, dx2, dy2 = doc['bbox']
            # 如果白色比例显著，标记为医生
            if doc['white_ratio'] > 0.15:
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (160, 160, 160), 2)  # 灰色框
                cv2.putText(frame, "Doctor", (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)

    cv2.imshow('Improved Tracking', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()