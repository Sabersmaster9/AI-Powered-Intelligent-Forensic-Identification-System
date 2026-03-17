import cv2
import mmcv
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmdet.apis import inference_detector, init_detector
from mmpose.visualization import PoseLocalVisualizer
import os

# 1. 注册 MMPose 所有模块
register_all_modules()

# --- 配置区 ---
device = 'cuda:0'  # 如果没有显卡，请改为 'cpu'
input_video = 'E:/Users/a/Desktop/Patient_Pose_Project/data/test1.mp4'  # 你的输入视频文件名
output_video = 'patient_analysis_result.mp4'  # 输出视频文件名

# 使用模型别名 (Model Alias)，MIM 会自动下载配置
# 检测器：使用 RTMDet-m (针对 COCO 数据集)
det_config = 'rtmdet_m_8xb32-300e_coco'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-b164563e.pth'

# 姿态估计：使用 HRNet-w32 (标准 17 个关键点)
pose_config = 'td-hm_hrnet-w32_8xb64-210e_coco-256x192'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-8160cbf5_20220913.pth'

# 2. 初始化模型
print("正在加载模型，请稍候...")
detector = init_detector(det_config, det_checkpoint, device=device)
pose_estimator = init_model(pose_config, pose_checkpoint, device=device)

# 3. 初始化可视化器
visualizer = PoseLocalVisualizer()
visualizer.dataset_meta = pose_estimator.dataset_meta

# 4. 视频处理流
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, size)

print(f"开始处理视频: {input_video} ...")

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- A. 目标检测 (找出所有人) ---
    det_result = inference_detector(detector, frame)
    pred_instances = det_result.pred_instances

    # 过滤：只保留置信度 > 0.5 的人 (类别 0 为人)
    bboxes = pred_instances.bboxes[
        (pred_instances.scores > 0.5) & (pred_instances.labels == 0)
        ].cpu().numpy()

    # --- B. 姿态估计 ---
    # Top-down 方法：将检测到的 bbox 传入
    pose_results = inference_topdown(pose_estimator, frame, bboxes)

    # --- C. 绘制连线与点 ---
    # 将结果叠加在当前帧
    visualizer.add_datasample(
        'result',
        frame,
        data_sample=pose_results[0] if len(pose_results) > 0 else None,
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=True,
        show=False
    )

    # 获取渲染后的 BGR 图像用于显示和保存
    output_frame = visualizer.get_image()

    # 写入文件并实时预览
    video_writer.write(output_frame)
    cv2.imshow('Patient Pose Analysis', output_frame)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"已处理 {frame_idx} 帧...")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 释放资源
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"处理完成！结果已保存至: {output_video}")