import time

import cv2
import glfw
import numpy as np
from loguru import logger
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# GLSLシェーダーコード
vertex_shader_code = """
#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoords;
out vec2 TexCoords;
void main() {
    TexCoords = texCoords;  // y座標の反転をやめる
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


fragment_shader_code = """
#version 330 core
in vec2 TexCoords;
out vec4 color;
uniform sampler2D ourTexture;
void main() {
    color = texture(ourTexture, TexCoords);
}
"""


# テクスチャ設定用関数
def create_texture():
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)  # 高速化
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    return texture_id


def main():
    # GLFW初期化
    if not glfw.init():
        return

    # ウィンドウ設定
    window = glfw.create_window(1280, 720, "OpenGL Video Playback", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(0)  # V-Sync無効

    # シェーダープログラムのコンパイル
    shader = compileProgram(
        compileShader(vertex_shader_code, GL_VERTEX_SHADER),
        compileShader(fragment_shader_code, GL_FRAGMENT_SHADER),
    )

    # 頂点データ
    vertices = np.array(
        [
            #  左下       tex=(0,1)
            -1.0,
            -1.0,
            0.0,
            1.0,
            #  右下       tex=(1,1)
            1.0,
            -1.0,
            1.0,
            1.0,
            #  左上       tex=(0,0)
            -1.0,
            1.0,
            0.0,
            0.0,
            #  右上       tex=(1,0)
            1.0,
            1.0,
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )

    # VAOとVBOの設定
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # 頂点属性ポインタ
    glVertexAttribPointer(
        0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0)
    )
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * vertices.itemsize,
        ctypes.c_void_p(2 * vertices.itemsize),
    )
    glEnableVertexAttribArray(1)

    # テクスチャの生成
    texture_id = create_texture()

    # OpenCVで動画を読み込み
    cap = cv2.VideoCapture("../output_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # メインループ
    data = []
    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        #######################
        # OpenGLテクスチャに転送 #
        #######################

        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_BGR, GL_UNSIGNED_BYTE, frame
        )

        #######
        # 描画 #
        #######

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader)
        glBindVertexArray(VAO)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glFinish()

        elapsed = time.perf_counter() - t0
        data.append(elapsed)
        logger.info(f"Elapsed time: {elapsed*1000:.4f} ms")

        # バッファのスワップ
        glfw.swap_buffers(window)
        glfw.poll_events()

    cap.release()
    glfw.terminate()
    # パフォーマンス計測の結果を表示
    print("Mean time: {:.4f} ms".format(np.mean(data[100:]) * 1000))
    print("Std time: {:.4f} ms".format(np.std(data[100:]) * 1000))


if __name__ == "__main__":
    main()
