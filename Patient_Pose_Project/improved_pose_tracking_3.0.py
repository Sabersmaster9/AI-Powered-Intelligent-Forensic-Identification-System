import cv2
import os
import numpy as np
from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# 注册 MMPose 模块
register_all_modules()

# ==================== 【关键配置区】 ====================

# 1. 颜色设定 (根据你提供的深色/黑色值)
DOCTOR_WHITE_RGB = [170, 164, 170]  # 医生白大褂的 RGB (带冷色调的白)
PATIENT_BLACK_RGB = [31, 30, 35]  # 病人黑色上衣的 RGB

# 2. 灵敏度与门槛
DOCTOR_TOL = 60  # 医生颜色容差
PATIENT_TOL = 30  # 黑色判定建议容差小一点，防止抓到深灰色
DOC_AREA_THRESH = 0.10  # 医生判定门槛 (10%)
PATIENT_AREA_THRESH = 0.15  # 病人判定门槛 (上半身黑色占比需 > 15%)

# =======================================================

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'weights', 'yolo26n.pt')
POSE_CONFIG = os.path.join(BASE_DIR, 'weights', 'rtmpose-m_config.py')
POSE_CHECKPOINT = os.path.join(BASE_DIR, 'weights', 'rtmpose-m.pth')
VIDEO_INPUT = os.path.join(BASE_DIR, 'data', 'test3.mp4')
VIDEO_OUTPUT = os.path.join(BASE_DIR, 'outputs', 'final_result.mp4')

# 骨架定义
COCO_17_INDICES = list(range(17))
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]


def get_color_score(crop, target_rgb, tolerance, part='full'):
    """
    计算颜色得分
    part='upper': 只计算上半部分 (避开裤子和鞋)
    """
    if crop.size == 0: return 0

    # 如果指定检测上半身
    if part == 'upper':
        h_roi, _, _ = crop.shape
        crop = crop[0:int(h_roi * 0.6), :]  # 取人体框的前 60% 区域

    target_bgr = np.uint8([[[target_rgb[2], target_rgb[1], target_rgb[0]]]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower = np.array([max(0, target_hsv[0] - 20), max(0, target_hsv[1] - tolerance), max(0, target_hsv[2] - tolerance)])
    upper = np.array(
        [min(180, target_hsv[0] + 20), min(255, target_hsv[1] + tolerance), min(255, target_hsv[2] + tolerance)])

    mask = cv2.inRange(hsv_crop, lower, upper)
    return np.sum(mask > 0) / (crop.shape[0] * crop.shape[1])


# --- 初始化模型 ---
det_model = YOLO(YOLO_PATH)
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device='cuda:0')

cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print("系统启动：正在精准定位黑衣病人...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = det_model(frame, verbose=False)
    candidates = []

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[max(0, y1):y2, max(0, x1):x2]

                # 病人只看上半身的黑色程度
                p_score = get_color_score(roi, PATIENT_BLACK_RGB, PATIENT_TOL, part='upper')
                # 医生看全身的白色程度
                d_score = get_color_score(roi, DOCTOR_WHITE_RGB, DOCTOR_TOL, part='full')

                candidates.append({
                    'bbox': [x1, y1, x2, y2],
                    'p_score': p_score,
                    'd_score': d_score
                })

    if candidates:
        # 1. 寻找潜在病人：按上半身黑色占比排序
        candidates.sort(key=lambda x: x['p_score'], reverse=True)
        best_p = candidates[0]

        # 标记病人是否被找到
        patient_found = False

        # 只有最高分通过门槛，才激活病人模式
        if best_p['p_score'] > PATIENT_AREA_THRESH:
            patient_found = True
            px1, py1, px2, py2 = best_p['bbox']

            # 姿态估计
            p_bbox_in = np.array([[px1, py1, px2, py2]], dtype=np.float32)
            pose_res = inference_topdown(pose_model, frame, p_bbox_in)

            # 绘图逻辑
            inst = pose_res[0].pred_instances
            kpts, k_scores = inst.keypoints[0], inst.keypoint_scores[0]
            for idx in COCO_17_INDICES:
                if k_scores[idx] > 0.3:
                    cv2.circle(frame, (int(kpts[idx][0]), int(kpts[idx][1])), 5, (0, 255, 0), -1)
            for s_idx, e_idx in SKELETON_17:
                if k_scores[s_idx] > 0.3 and k_scores[e_idx] > 0.3:
                    cv2.line(frame, (int(kpts[s_idx][0]), int(kpts[s_idx][1])),
                             (int(kpts[e_idx][0]), int(kpts[e_idx][1])), (0, 255, 0), 2)

            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, f"Patient ({best_p['p_score']:.2f})", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 2. 处理医生（跳过已确定的病人候选人）
        start_idx = 1 if patient_found else 0
        for doc in candidates[start_idx:]:
            if doc['d_score'] > DOC_AREA_THRESH:
                dx1, dy1, dx2, dy2 = doc['bbox']
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (160, 160, 160), 2)
                cv2.putText(frame, "Doctor", (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)

    out.write(frame)
    cv2.imshow('ZJU Intelligent Sports - Work Injury Assessment', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()