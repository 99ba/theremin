# 手势特雷门琴
这是一个基于摄像头手部识别的实时交互式电子乐器项目。程序使用 OpenCV 显示界面和读取摄像头，使用 MediaPipe Hand Landmarker 识别双手关键点，并通过实时音频合成器生成旋律、和弦、节拍器和演奏录音。

项目当前包含两个入口：

- `Basic`：面向基础演示和曲谱跟奏，包含自由演奏、手势学习和带曲谱引导的 `Hybrid 1`。
- `Professional`：面向完整演奏展示，包含手势学习、专业音域设置、`Hybrid 1` 和 `Hybrid 2`。

独立的 `Gesture Play` 和独立的 `Trajectory Guide` 当前没有出现在模式菜单中。曲谱引导功能集成在 Basic `Hybrid 1` 中，手势识别演奏功能集成在 `Hybrid 1` / `Hybrid 2` 中。

## 快速运行

在 PowerShell 中进入项目目录：

```powershell
cd "C:\Users\33525\Desktop\改进\theremin"
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行程序：

```powershell
python main.py
```

如果使用本机已有的 Conda 环境，可以直接运行：

```powershell
& "F:\Anaconda\envs\p1\python.exe" main.py
```

当前代码已用该环境做过语法编译和依赖导入检查。

## 依赖

`requirements.txt` 当前包含：

| 依赖 | 用途 |
| --- | --- |
| `opencv-python` | 摄像头读取、窗口 UI、图像绘制、视频写入 |
| `mediapipe` | 手部关键点识别 |
| `numpy` | 手势特征、音频缓冲、数值计算 |
| `sounddevice` | 实时音频输出 |
| `scikit-learn` | 静态手势 RBF-SVM 分类器 |
| `joblib` | 静态手势模型保存和加载 |
| `imageio-ffmpeg` | 找不到系统 FFmpeg 时用于音视频合并 |
| `torch` | 动态手势 GRU 模型训练和推理 |

注意：当前仓库内的 `models/static_gesture_svm.joblib` 是随项目保存的模型文件。若本机 `scikit-learn` 版本和训练时版本不同，加载时可能出现版本警告。遇到识别不稳定时，建议在当前环境中重新训练静态手势模型。

## 启动流程

1. 运行 `main.py`。
2. 在首页选择 `Basic` 或 `Professional`。
3. 进入预览页面后点击右下角菜单按钮。
4. 选择模式。
5. 根据模式完成曲目选择、专业音域设置或手势录制准备。
6. 在准备页点击 `Start` 或按 `SPACE` 开始。

常用按键：

| 按键 | 功能 |
| --- | --- |
| `SPACE` | 在准备页开始演奏；演奏中暂停或继续 |
| `ESC` | 返回上一页；演奏中退出到菜单 |
| `q` / `Q` | 退出程序 |

## 核心控制方式

### 音高

当前连续旋律音高由物理右手控制：

```text
物理右手食指到 Pitch Anchor 的距离越近 -> 音高越高
物理右手食指到 Pitch Anchor 的距离越远 -> 音高越低
```

相关代码：

- `app/hand_features.py` 生成 `right_distance_to_anchor`。
- `app/pitch_mapper.py` 将距离映射为 MIDI 音高。
- `app/controller.py` 和 `app/mixure_enhanced_controller.py` 生成最终音频控制信号。

### 音量

当前所有主要演奏模式的旋律音量统一由物理左手在画面中的上下位置控制：

```text
左手在画面上方 -> 高音量
左手在画面中间 -> 中音量
左手在画面下方 -> 低音量
未检测到左手 -> 静音
```

默认三档配置在 `config.py` 中：

```python
LEFT_VOLUME_TOP = 0.80
LEFT_VOLUME_MID = 0.56
LEFT_VOLUME_BOTTOM = 0.32
LEFT_VOLUME_TOP_ZONE_RATIO = 0.33
LEFT_VOLUME_MID_ZONE_RATIO = 0.66
```

物理右手的张开 / 捏合仍用于发声门控或触发，不再作为主音量来源。

### 左右手职责

在 `Hybrid 1` 中：

```text
物理右手 -> 主旋律音高 + 发声门控
物理左手 -> 音量位置 + 手势和弦/单音触发
```

在 `Hybrid 2` 中：

```text
物理右手 -> 手势旋律
物理左手 -> 音量位置 + 手势伴奏
```

左手同时承担“上下位置控制音量”和“手型识别伴奏”的职责，因此演奏时需要让左手保持在目标音量区域，同时做出可识别的手势形状。

## Basic 模式

Basic 当前显示以下模式：

- `Free play`
- `Gesture learning`
- `Hybrid 1`

### Free Play

自由演奏模式，不依赖曲谱或手势模板。

主要用于测试摄像头、音频输出、右手音高映射、左手音量控制和基础演奏手感。

### Gesture Learning

手势录制和管理模式。支持：

- 录制静态手势。
- 录制动态手势。
- 选择手势属于左手或右手。
- 给手势绑定单音或和弦。
- 删除已有手势。
- 自动训练静态 SVM 模型。
- 自动尝试训练动态 GRU 模型。

手势绑定示例：

```text
C4
D#4
Bb3
F#4
C
Cm
C7
Cmaj7
Am7
```

相关文件：

```text
assets/gesture_templates.json
assets/dynamic_gesture_templates.json
models/static_gesture_svm.joblib
assets/dynamic_gesture_gru.pt
logs/dynamic_recording_debug.log
```

当前仓库内已有静态模板主要是左手模板。若要让 Professional `Hybrid 2` 的右手静态手势稳定工作，需要录制至少两个右手静态手势并重新训练模型。

### Hybrid 1

Basic `Hybrid 1` 是当前基础版的主要演示模式，集成曲谱引导、右手旋律、左手音量和左手手势伴奏。

进入流程：

1. 选择 `Basic`。
2. 进入 `Hybrid 1`。
3. 选择曲目。
4. 进入准备页。
5. 可选择 `Record ON`。
6. 点击 `Start` 或按 `SPACE`。
7. 根据提示完成右手三点音域校准。
8. 进入正式演奏。

演奏控制：

- 右手食指跟随曲谱轨迹控制音高。
- 右手张开 / 捏合控制旋律发声。
- 左手上 / 中 / 下位置控制旋律音量。
- 左手手势触发绑定的单音或和弦。

演奏中左上角控制菜单支持：

| 控件 | 功能 |
| --- | --- |
| `Menu` | 展开或收起控制菜单 |
| `Piano` | 切换基础版钢琴单次触发效果 |
| `Song -` / `Song +` | 切换曲目 |
| `Restart` | 当前曲目从头开始 |
| `Pause` | 暂停或继续曲谱 |
| `Speed -` / `Speed +` | 调整曲谱速度 |
| `Beat` | 打开或关闭节拍提示 |

Basic 曲目来自 `app/mixure_guide_track.py` 的内置曲库和 `scores/*.json` 的外部曲谱。当前包含：

- `Liangzhu`
- `Twinkle Star`
- `Traumerei`
- `Canghai`
- `Songbie`
- `Songbie Uploaded`

## Professional 模式

Professional 当前显示以下模式：

- `Gesture learning`
- `Hybrid 1`
- `Hybrid 2`

Professional 不显示 `Free play`。若需要自由测试，请使用 Basic `Free play`。

### Professional Setup

进入 Professional `Hybrid 1` 或 `Hybrid 2` 前，会先进入专业设置页。

可设置：

| 项目 | 说明 |
| --- | --- |
| `Low note` | 专业音域最低音，例如 `C4` |
| `High note` | 专业音域最高音，例如 `G4` |
| `Timbre` | 音色预设 |

音色选项：

| 音色 | 说明 |
| --- | --- |
| `Sustain Piano` | 连续型钢琴音色 |
| `Mixure Piano` | 单次触发并带延音的钢琴音色 |
| `Clarinet` | 连续单簧管音色 |

专业音域会被转换为大调音阶音符，并写入 `CUSTOM_SCALE_NOTES`。右手旋律会吸附到这些可用音符上。

### Professional Hybrid 1

适合展示“右手连续旋律 + 左手手势伴奏”。

控制方式：

- 右手食指到 `Pitch Anchor` 的距离控制音高。
- 右手张开 / 捏合控制旋律发声或触发。
- 左手上 / 中 / 下位置控制旋律音量。
- 左手静态或动态手势触发单音 / 和弦伴奏。

### Professional Hybrid 2

适合展示“双手手势触发”。

控制方式：

- 右手手势控制旋律音符。
- 左手手势控制伴奏或和弦。
- 左手上 / 中 / 下位置控制整体旋律音量。
- 如果右手手势绑定音符不在专业音域中，会吸附到最近的可用音符。
- 如果启用了 fallback 且检测到右手但没有识别出旋律手势，会使用 fallback 音符。

当前注意事项：

- 仓库内现有静态 SVM 模型只训练出了左手模型。
- 右手静态手势模板不足时，`Hybrid 2` 的右手静态手势不会稳定工作。
- 需要通过 `Gesture learning` 录制右手静态或动态手势后再使用 `Hybrid 2`。

## 曲谱导入

Basic `Hybrid 1` 支持读取 `scores/*.json` 中的外部曲谱。

当前外部曲谱格式由：

```text
scores/player_guide_schema.json
```

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

这些脚本通过命令行参数或环境变量读取 API Key，仓库中不应提交真实密钥。

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

## 项目结构

```text
theremin/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── ui.png
├── app/
│   ├── camera.py
│   ├── hand_tracker.py
│   ├── hand_features.py
│   ├── controller.py
│   ├── mixure_enhanced_controller.py
│   ├── gesture_classifier.py
│   ├── ui_manager.py
│   ├── renderer.py
│   ├── audio_engine.py
│   ├── synth.py
│   ├── mixure_articulation_synth.py
│   ├── mixure_guide_track.py
│   ├── mixure_enhanced_guide_track.py
│   ├── static_svm.py
│   ├── dynamic_gru.py
│   └── performance_recorder.py
├── assets/
│   ├── hand_landmarker.task
│   ├── hand_outline.png
│   ├── hand_outline_source.png
│   ├── gesture_templates.json
│   ├── dynamic_gesture_templates.json
│   └── dynamic_gesture_gru.pt
├── models/
│   └── static_gesture_svm.joblib
├── scores/
│   ├── player_guide_schema.json
│   └── songbie_uploaded.json
├── tools_mixure/
├── logs/
├── recordings/
└── datasets/
```

说明：

- `assets/hand_landmarker.task` 是 MediaPipe 手部识别模型，运行必需。
- `assets/gesture_templates.json` 是静态手势模板。
- `assets/dynamic_gesture_templates.json` 是动态手势模板。
- `models/static_gesture_svm.joblib` 是静态手势分类模型。
- `datasets/` 是本地采集数据，默认不建议提交到公开仓库。
- `logs/` 和 `recordings/` 是运行输出，默认不建议提交。

## 重要配置

主要配置位于 `config.py`。

### 摄像头

```python
CAMERA_INDEX
FRAME_WIDTH
FRAME_HEIGHT
CAMERA_FPS
CAMERA_BUFFER_SIZE
FLIP_HORIZONTAL
SHOW_HANDEDNESS_DEBUG
```

当前主入口会根据 `FLIP_HORIZONTAL` 对自拍式画面做左右手标准化。`SWAP_HANDEDNESS` 在配置中保留，但主入口当前没有传入 `HandTracker`，因此它不是主要调试开关。

### 音高和音量

```python
RIGHT_DISTANCE_MIN
RIGHT_DISTANCE_MAX
PITCH_DISTANCE_CURVE
LEFT_VOLUME_TOP_ZONE_RATIO
LEFT_VOLUME_MID_ZONE_RATIO
LEFT_VOLUME_TOP
LEFT_VOLUME_MID
LEFT_VOLUME_BOTTOM
```

### Basic Hybrid 1

```python
BASIC_HYBRID1_DEFAULT_SONG
BASIC_HYBRID1_SONG_SPEED_MIN
BASIC_HYBRID1_SONG_SPEED_MAX
BASIC_HYBRID1_SONG_SPEED_STEP
BASIC_HYBRID1_GUIDE_ASSIST_STRENGTH
BASIC_HYBRID1_ENABLE_SCORING
BASIC_HYBRID1_SYNTH_MODE
BASIC_PITCH_CALIBRATION_ENABLED
BASIC_PITCH_CALIBRATION_HOLD_SECONDS
```

### Professional

```python
PRO_PITCH_LOW_NOTE
PRO_PITCH_HIGH_NOTE
PRO_PITCH_STEP_SEMITONES
PRO_PITCH_RING_ENABLED
PRO_TIMBRE_PRESET
```

### 手势识别

```python
STATIC_GESTURE_MODEL_PATH
STATIC_GESTURE_SAMPLES_PER_CLASS_MIN
STATIC_GESTURE_RECORD_SECONDS
STATIC_GESTURE_MIN_CONFIDENCE
STATIC_GESTURE_MARGIN
DYNAMIC_GESTURE_WINDOW_FRAMES
DYNAMIC_GESTURE_RECORD_SECONDS
DYNAMIC_GESTURE_MIN_CONFIDENCE
DYNAMIC_GESTURE_USE_GRU
```

### 演奏录制

```python
PERFORMANCE_RECORD_DIR
PERFORMANCE_RECORD_FPS
```

## 上传 GitHub 前建议

仓库已包含 `.gitignore`，默认排除：

- `__pycache__/`
- `.DS_Store`
- `logs/`
- `recordings/`
- `datasets/`
- `.env`
- 视频和 WAV 导出文件

建议提交运行必需的代码和资产，谨慎提交本地采集图片数据。`datasets/` 中可能包含个人手部和环境画面，公开上传前需要确认隐私风险。

## 常见问题

### 没有声音

检查系统默认音频输出设备，并确认 `sounddevice` 可用。程序启动时如果音频设备不可用，会显示提示，但视觉界面仍可运行。

### 摄像头打不开

修改 `config.py` 中的：

```python
CAMERA_INDEX
```

常见摄像头编号为 `0` 或 `1`。

### 左右手反了

检查：

```python
FLIP_HORIZONTAL
```

当前项目为了自拍式显示会镜像摄像头画面，并在 `main.py` 中按 `FLIP_HORIZONTAL` 交换左右手控制数据。`SWAP_HANDEDNESS` 虽然存在于 `config.py`，但主入口当前没有传入 `HandTracker`。

### Hybrid 2 右手手势没有反应

当前仓库内静态手势模板主要是左手模板。进入 `Gesture learning`，录制至少两个右手静态手势，或录制右手动态手势，然后重新进入 `Hybrid 2`。

### 录像没有合并声音

确认系统中有 `ffmpeg`，或已安装 `imageio-ffmpeg`。即使合并失败，WAV 音频文件仍会保存在 `recordings/` 中。

## 验证命令

语法编译检查：

```powershell
& "F:\Anaconda\envs\p1\python.exe" -m compileall -q .
```

依赖导入检查：

```powershell
& "F:\Anaconda\envs\p1\python.exe" -c "import cv2, mediapipe, numpy, sounddevice, sklearn, joblib, imageio_ffmpeg, torch; print('deps ok')"
```
