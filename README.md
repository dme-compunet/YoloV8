# YOLOv8

Use [YOLOv8](https://github.com/ultralytics/ultralytics) in real-time for object detection, instance segmentation, pose estimation and image classification, via [ONNX Runtime](https://github.com/microsoft/onnxruntime)

# Install

The `YoloV8` project is available in two nuget packages: [YoloV8](https://www.nuget.org/packages/YoloV8) and [YoloV8.Gpu](https://www.nuget.org/packages/YoloV8.Gpu), if you use with CPU add the [YoloV8](https://www.nuget.org/packages/YoloV8) package reference to your project (contains reference to [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) package)

```shell
dotnet add package YoloV8
```

If you use with GPU you can add the [YoloV8.Gpu](https://www.nuget.org/packages/YoloV8.Gpu) package reference (contains reference to [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu) package)

```shell
dotnet add package YoloV8.Gpu
```

# Use

### Export the model from PyTorch to ONNX format:

Run this python code to export the model in ONNX format:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('path/to/best')

# export the model to ONNX format
model.export(format='onnx')
```

### Use in exported model with C#:

```csharp
using Compunet.YoloV8;
using SixLabors.ImageSharp;

using var predictor = YoloV8Predictor.Create("path/to/model");

var result = predictor.Detect("path/to/image");
// or
var result = await predictor.DetectAsync("path/to/image");

Console.WriteLine(result);
```

# Plotting

You can to plot the input image for preview the model prediction results, this code demonstrates how to perform a prediction, plot the results and save to file:

```csharp
using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;
using SixLabors.ImageSharp;

using var image = Image.Load("path/to/image");

using var predictor = YoloV8Predictor.Create("path/to/model");

var result = await predictor.PoseAsync(image);

using var plotted = await result.PlotImageAsync(image);

plotted.Save("./pose_demo.jpg")
```

You can also predict and save to file in one operation:

```csharp
using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;
using SixLabors.ImageSharp;

using var predictor = YoloV8Predictor.Create("path/to/model");

predictor.PredictAndSaveAsync("path/to/image");
```

## Demo Images:

#### Detection:

![detect-demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/Assets/detect-demo.jpg)

#### Pose:

![pose-demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/Assets/pose-demo.jpg)

#### Segmentation:

![seg-demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/Assets/seg-demo.jpg)

# License

MIT License