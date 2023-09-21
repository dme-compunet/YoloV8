# YOLOv8

Use [YOLOv8](https://github.com/ultralytics/ultralytics) in real-time for object detection, instance segmentation, pose estimation and image classification, via [ONNX Runtime](https://github.com/microsoft/onnxruntime)

# Install

The `YoloV8` project is available in two versions of nuget packages: [YoloV8](https://www.nuget.org/packages/YoloV8) and [YoloV8.Gpu](https://www.nuget.org/packages/YoloV8.Gpu), if you use with CPU add the [YoloV8](https://www.nuget.org/packages/YoloV8) package reference to your project (contains reference to [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) package)

```shell
dotnet add package YoloV8 --version 1.6.0
```

If you use with GPU you need to add the [YoloV8.Gpu](https://www.nuget.org/packages/YoloV8.Gpu) package reference (contains reference to [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu) package)

```shell
dotnet add package YoloV8.Gpu --version 1.6.0
```

# Use

### Export the model from PyTorch to ONNX format:

Run the following python code to export the model to ONNX format:

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

using var predictor = new YoloV8(model);

var result = predictor.Detect("path/to/image");
// or
var result = await predictor.DetectAsync("path/to/image");

Console.WriteLine(result);
```

# Plotting

You can to plot the input image for preview the model prediction results, this code demonstrates how to perform a prediction with the model and then plot the prediction results on the input image and save to file:

```csharp
using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;
using SixLabors.ImageSharp;

var imagePath = "path/to/image";

using var predictor = new YoloV8("path/to/model");

var result = predictor.Pose(imagePath);

using var image = Image.Load(imagePath);
using var ploted = result.PlotImage(image);

ploted.Save("./pose_demo.jpg")
```

## Demo Images:

#### Detection:

![detect_demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/assets/detect_demo.jpg)

#### Pose:

![pose_demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/assets/pose_demo.jpg)

#### Segmentation:

![seg_demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/assets/seg_demo.jpg)

# License

MIT License
