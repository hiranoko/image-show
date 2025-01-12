#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <opencv2/core/cuda.hpp>
#include <vector>
#include <chrono>

// ビルド情報を表示する関数
void printBuildInformation() {
    std::string build_info = cv::getBuildInformation();
    std::cout << "OpenCV Build Information:" << std::endl;
    std::cout << build_info << std::endl;

    // OpenCVがCUDAをサポートしているか確認
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "OpenCV is built with CUDA support." << std::endl;

        // CUDA対応デバイス数を表示
        int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "Number of CUDA-enabled devices: " << deviceCount << std::endl;

        // 各デバイスの情報を表示
        for (int i = 0; i < deviceCount; ++i) {
            cv::cuda::DeviceInfo deviceInfo(i);
            std::cout << "Device " << i << ": " << deviceInfo.name() << std::endl;
            std::cout << "  Compute capability: " << deviceInfo.majorVersion() << "." << deviceInfo.minorVersion() << std::endl;
            std::cout << "  Total memory: " << deviceInfo.totalMemory() / (1024 * 1024) << " MB" << std::endl;
        }
    } else {
        std::cout << "OpenCV is NOT built with CUDA support, or no CUDA-enabled devices are available." << std::endl;
    }
}

int main()
{
    // 動画を読み込む
    cv::VideoCapture cap("../../output_video.mp4");
    if (!cap.isOpened())
    {
        fmt::print("Error: Could not open video.\n");
        return -1;
    }

    // 動画情報の取得
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    fmt::print("Video resolution: {}x{}, FPS: {:.2f}, Total frames: {}\n", width, height, fps, total_frames);

    cv::namedWindow("Display", cv::WINDOW_NORMAL);
    // cv::namedWindow("Display", cv::WINDOW_OPENGL);

    std::vector<double> times; // 時間計測用

    while (true)
    {
        cv::Mat frame;

        // フレームを取得
        bool ret = cap.read(frame);
        if (!ret)
        {
            break; // 動画終了
        }

        // タイマー開始
        auto t0 = std::chrono::high_resolution_clock::now();

        // フレームを描画
        cv::imshow("Display", frame);

        // ユーザー入力待機 ('q' or 'ESC' で終了)
        char key = static_cast<char>(cv::pollKey());

        // タイマー終了
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        times.push_back(elapsed.count());

        fmt::print("Elapsed time: {:.4f} ms\n", elapsed.count() * 1000);

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }

    // パフォーマンスの結果を表示
    if (times.size() > 100)
    {
        double sum = 0.0;
        double sq_sum = 0.0;
        int skip = 100;

        for (size_t i = skip; i < times.size(); ++i)
        {
            sum += times[i];
            sq_sum += times[i] * times[i];
        }

        double mean = sum / (times.size() - skip);
        double stddev = std::sqrt(sq_sum / (times.size() - skip) - mean * mean);

        fmt::print("Mean time: {:.4f} ms\n", mean * 1000);
        fmt::print("Std time: {:.4f} ms\n", stddev * 1000);
    }

    cap.release();
    cv::destroyAllWindows();

    // OpenCVのビルド情報を表示
    printBuildInformation();

    
    return 0;
}
