import time

import cv2
import numpy as np
from loguru import logger


def main():
    # 動画キャプチャの初期化
    cap = cv2.VideoCapture("../output_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数

    logger.info(f"Video resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    cv2.namedWindow("imshow", cv2.WINDOW_NORMAL)

    data = []
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            t0 = time.perf_counter()
            cv2.imshow("imshow", frame)
            ch = cv2.pollKey()
            elapsed = time.perf_counter() - t0
            data.append(elapsed)
            logger.info(f"Elapsed time: {elapsed*1000:.4f} ms")

            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        else:
            break
    # パフォーマンス計測の結果を表示
    print("Mean time: {:.4f} ms".format(np.mean(data[100:]) * 1000))
    print("Std time: {:.4f} ms".format(np.std(data[100:]) * 1000))


if __name__ == "__main__":
    main()
