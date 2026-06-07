<<<<<<< HEAD
# Gesture Theremin

一个基于摄像头手势识别的实时电子乐器。它使用 MediaPipe 识别双手关键点，通过右手控制旋律音高、左手控制音量和伴奏手势，让普通摄像头变成一个可演奏、可录制、可自定义手势的虚拟 Theremin。
=======
# 手势特雷门琴
这是一个基于摄像头手部识别的实时交互式电子乐器项目。程序使用 OpenCV 显示界面和读取摄像头，使用 MediaPipe Hand Landmarker 识别双手关键点，并通过实时音频合成器生成旋律、和弦、节拍器和演奏录音。
>>>>>>> ff6246b135708a24d3eed7187bf2ef7d9d780799

![Gesture Theremin UI](ui.png)

## 功能亮点

- 实时摄像头手部追踪：识别左右手关键点，并在画面中显示演奏状态。
- 右手旋律控制：通过右手食指与音高锚点的距离控制音高。
- 左手音量控制：左手在画面上、中、下区域切换不同音量层级。
- 自定义手势学习：录制静态手势或动态手势，并绑定到单音或和弦。
- 曲谱引导演奏：内置曲目轨迹提示，支持跟随曲谱进行演奏。
- 专业演奏模式：可设置音域、音色，并使用双手手势完成旋律与伴奏组合。
- 演出录制：可保存带 UI 的演奏视频、纯摄像头视频以及分轨音频。
- 曲谱导入工具：支持将外部曲谱 JSON 导入到 `scores/` 中使用。

## 演奏模式

### Basic

适合快速体验和基础演示。

- `Free play`：自由演奏，不依赖曲谱或手势模板。
- `Gesture learning`：录制和管理自定义手势。
- `Hybrid 1`：右手控制旋律，左手控制音量与手势伴奏，并可跟随曲谱轨迹演奏。

### Professional

适合更完整的双手演奏展示。

- `Gesture learning`：录制静态或动态手势，并绑定音符或和弦。
- `Hybrid 1`：右手连续控制旋律，左手用手势触发伴奏。
- `Hybrid 2`：右手手势触发旋律，左手手势触发伴奏或和弦。

Professional 模式进入演奏前可以设置：

- `Low note` / `High note`：演奏音域。
- `Timbre`：音色预设，包括钢琴类音色和单簧管类连续音色。

## 快速开始

本项目需要摄像头和音频输出设备。建议使用 Python 3.10+ 的虚拟环境：

```powershell
git clone <your-repo-url>
cd theremin
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

如果你已经有可用的 Python 环境，也可以直接运行：

```powershell
python -m pip install -r requirements.txt
python main.py
```

## 基本操作

1. 启动 `main.py`。
2. 选择 `Basic` 或 `Professional`。
3. 进入预览页面后打开菜单。
4. 选择演奏模式或 `Gesture learning`。
5. 按页面提示完成曲目选择、音域设置、手势录制或演奏准备。
6. 点击 `Start` 或按 `SPACE` 开始。

常用按键：

| 按键 | 功能 |
| --- | --- |
| `SPACE` | 开始演奏；演奏中暂停或继续 |
| `ESC` | 返回上一页；演奏中退出到菜单 |
| `q` / `Q` | 退出程序 |

## 手势录制建议

进入 `Gesture learning` 后，可以创建新的静态手势或动态手势，并把它绑定到单音或和弦，例如 `C4`、`D#4`、`Bb3`、`C`、`Cm`、`C7`、`Am7`。

录制时请注意：

- 动态手势可以选择录制多轮，建议多选几轮以提高识别稳定性。
- 录制过程中不要完全定住手，保持手势形状的同时让手在画面中轻微移动。
- 同一个手势最好覆盖不同位置和距离，避免模型只记住某一个固定画面位置。
- `Hybrid 2` 需要右手手势参与旋律触发，如果右手没有反应，请先录制右手静态或动态手势。

手势数据会保存到：

```text
assets/gesture_templates.json
assets/dynamic_gesture_templates.json
models/static_gesture_svm.joblib
assets/dynamic_gesture_gru.pt
```

## 演出录制

演奏准备页面可以切换：

```text
Record OFF / Record ON
```

开启后，退出演奏或返回菜单时会在 `recordings/` 中保存：

- 带 UI、轨迹和 HUD 的演奏视频。
- 纯摄像头画面视频。
- 完整混音音频。
- 旋律音轨。
- 和弦或伴奏音轨。

程序会优先使用系统 `ffmpeg` 合并音视频；如果系统没有安装，会尝试使用 `imageio-ffmpeg`。

## 曲谱

Basic `Hybrid 1` 支持内置曲目和 `scores/*.json` 中的外部曲谱。当前曲目包括：

- `Liangzhu`
- `Twinkle Star`
- `Traumerei`
- `Canghai`
- `Songbie`
- `Songbie Uploaded`

曲谱导入相关工具位于 `tools_mixure/`。外部曲谱格式可参考：

```text
scores/player_guide_schema.json
```

<<<<<<< HEAD
=======
说明。导入工具位于：

```text
tools_mixure/
```

相关脚本：

| 脚本 | 用途 |
| --- | --- |
| `convert_jianpu_images_qwenvl.py` | 使用视觉模型将简谱图片转换为中间 JSON |
| `convert_jianpu_images_deepseek.py` | 兼容命名的图片转换脚本 |
| `import_player_guide_json.py` | 将中间 JSON 导入为项目曲谱 |

这些脚本通过命令行参数或环境变量读取 API Key。

## 演奏录制

在演奏准备页可以切换：

```text
Record OFF / Record ON
```

开启后，退出演奏或返回菜单时会保存：

```text
recordings/<时间>_<版本>_<模式>_with_tracks.mp4
recordings/<时间>_<版本>_<模式>_person_only.mp4
recordings/<时间>_<版本>_<模式>_audio.wav
recordings/<时间>_<版本>_<模式>_melody.wav
recordings/<时间>_<版本>_<模式>_chord.wav
```

文件说明：

| 文件 | 内容 |
| --- | --- |
| `with_tracks.mp4` | 带 HUD、轨迹和 UI 的演奏画面 |
| `person_only.mp4` | 原始摄像头画面 |
| `audio.wav` | 完整混音 |
| `melody.wav` | 旋律音轨 |
| `chord.wav` | 和弦 / 伴奏音轨 |

程序会优先使用系统 `ffmpeg` 合并音频和视频。找不到时会尝试使用 `imageio-ffmpeg`。如果两者都不可用，视频仍会保存，但音频只保存在 WAV 文件中。

>>>>>>> ff6246b135708a24d3eed7187bf2ef7d9d780799
## 项目结构

```text
theremin/
|-- main.py
|-- config.py
|-- requirements.txt
|-- ui.png
|-- app/
|   |-- camera.py
|   |-- hand_tracker.py
|   |-- hand_features.py
|   |-- controller.py
|   |-- gesture_classifier.py
|   |-- gesture_recorder.py
|   |-- dynamic_gesture.py
|   |-- audio_engine.py
|   |-- synth.py
|   `-- performance_recorder.py
|-- assets/
|   |-- hand_landmarker.task
|   |-- gesture_templates.json
|   `-- dynamic_gesture_templates.json
|-- models/
|-- scores/
`-- tools_mixure/
```

## 依赖

主要依赖：

- `opencv-python`：摄像头、窗口 UI 和图像绘制。
- `mediapipe`：手部关键点识别。
- `numpy`：特征处理和音频计算。
- `sounddevice`：实时音频输出。
- `scikit-learn` / `joblib`：静态手势分类模型。
- `torch`：动态手势 GRU 模型。
- `imageio-ffmpeg`：音视频合并的备用方案。

## 常见问题

### 没有声音

检查系统默认音频输出设备，并确认 `sounddevice` 可用。音频设备不可用时，视觉界面仍可运行。

### 摄像头打不开

修改 `config.py` 中的 `CAMERA_INDEX`。常见摄像头编号为 `0` 或 `1`。

### 左右手识别相反

检查 `config.py` 中的 `FLIP_HORIZONTAL`。项目默认采用自拍式镜像画面。

### 手势识别不稳定

重新进入 `Gesture learning` 录制更多样本。录制动态手势时可以多选几轮，并在每轮录制时保持手势移动，让模型看到更多位置和距离变化。

## 验证

```powershell
python -m compileall -q .
python -c "import cv2, mediapipe, numpy, sounddevice, sklearn, joblib, imageio_ffmpeg, torch; print('deps ok')"
```
