import cv2
import numpy as np

def create_video(output_path, width=1280, height=720, fps=60, duration=10):
    # 動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデック
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 総フレーム数
    total_frames = fps * duration

    for frame_num in range(total_frames):
        # フレームを作成 (黒背景)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 数字を描画
        text = f"Frame: {frame_num + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        color = (255, 255, 255)  # 白
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # フレームを動画に書き込み
        video_writer.write(frame)

        # 進捗表示
        if (frame_num + 1) % 100 == 0:
            print(f"Processed {frame_num + 1}/{total_frames} frames")
    
    # リソース解放
    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    output_file = "output_video.mp4"
    create_video(output_file)
