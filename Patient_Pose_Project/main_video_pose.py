import cv2
import os
import numpy as np
from ultralytics import YOLO

# 保持 1.3.2 版本的底层导入路径
from mmpose.apis import init_model
from mmpose.apis.inference import inference_topdown
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer

# 强制注册
register_all_modules()

# --- 1. 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'weights', 'yolo26n.pt')
POSE_CONFIG = os.path.join(BASE_DIR, 'weights', 'rtmpose-m_config.py')
POSE_CHECKPOINT = os.path.join(BASE_DIR, 'weights', 'rtmpose-m.pth')
VIDEO_INPUT = os.path.join(BASE_DIR, 'data', 'test1.mp4')
VIDEO_OUTPUT = os.path.join(BASE_DIR, 'outputs', 'result_video.mp4')

# --- 2. 初始化模型 ---
print("正在加载 YOLO26 检测模型...")
det_model = YOLO(YOLO_PATH)

print("正在加载 MMPose 模型...")
# init_model 会把上面补全的 pipeline 加载进 pose_model.cfg
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device='cuda:0')

visualizer = PoseLocalVisualizer()
visualizer.set_dataset_meta(pose_model.dataset_meta)


def is_patient_by_color(crop):
    """HSV 空间颜色过滤：检测浅蓝色病号服"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ratio = np.sum(mask > 0) / (crop.shape[0] * crop.shape[1]) if crop.size > 0 else 0
    return ratio > 0.08


# --- 3. 视频处理 ---
cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

print(f"开始处理视频流...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 检测
    det_results = det_model(frame, verbose=False, stream=True)
    patient_bbox = None

    for r in det_results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if is_patient_by_color(crop):
                    patient_bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                    break

    # 姿态推理
    if patient_bbox is not None:
        # 核心：执行推理
        results = inference_topdown(pose_model, frame, patient_bbox)

        # 核心：手动渲染骨架
        visualizer.add_datasample(
            'result',
            frame,
            data_sample=results[0],
            draw_gt=False,
            show=False,
            kpt_thr=0.3
        )
        frame = visualizer.get_image()

        bx = patient_bbox[0].astype(int)
        cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Patient Locked", (bx[0], bx[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Final Pose Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("全部处理完毕！")