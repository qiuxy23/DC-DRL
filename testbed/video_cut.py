import cv2


video_path = 'videos/Batman_vs_Superman_Official_Teaser_Trailer_cut_1.mp4'
cap = cv2.VideoCapture(video_path)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 创建保存视频文件类对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(width), int(height)))

start_frame = 0
end_frame = 100

# 设置帧读取的开始位置
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
pos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # 获得帧位置
while (pos <= end_frame):
    ret, frame = cap.read()  # 捕获一帧图像
    out.write(frame)  # 保存帧
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

cap.release()
out.release()
