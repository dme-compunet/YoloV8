# YoloV8

Integrate [YOLOv8](https://github.com/ultralytics/ultralytics) into your C# project for a variety of real-time tasks including object detection, instance segmentation, pose estimation and more, using ONNX Runtime.

# Features
- **YOLOv8 Tasks** 🌟 Support for all YOLOv8 tasks ([Detect](https://docs.ultralytics.com/tasks/detect), [Segment](https://docs.ultralytics.com/tasks/segment), [Classify](https://docs.ultralytics.com/tasks/classify), [Pose](https://docs.ultralytics.com/tasks/pose) and [OBB](https://docs.ultralytics.com/tasks/obb))
- **High Performance** 🚀 Various techniques and use of .NET features to maximize performance
- **Reduced Memory Usage** 🧠 By reusing memory blocks and reducing the pressure on the GC
- **Plotting Options** 📊 Plotting operations for preview of model results on the target image.
- **YOLOv10 Support** 🔧 Includes additional support for [YOLOv10](https://docs.ultralytics.com/models/yolov10)

# Installation
This project provides two NuGet packages:
- For CPU inference, use the package: [YoloV8](https://www.nuget.org/packages/YoloV8) (includes the [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) package)
- For GPU inference, use the package: [YoloV8.Gpu](https://www.nuget.org/packages/YoloV8.Gpu) (includes the [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu) package)

# Usage

### 1. Export model to ONNX format:

For convert the pre-trained PyTorch model to ONNX format, run the following Python code:
```python
from ultralytics import YOLO

# Load a model
model = YOLO('path/to/best.pt')

# Export the model to ONNX format
model.export(format='onnx')
```

### 2. Load the ONNX model with C#:

Add the `YoloV8` (or `YoloV8.Gpu`) package to your project:
```shell
dotnet add package YoloV8
```

Use the following C# code to load the model and run basic prediction:
```csharp
using Compunet.YoloV8;

// Load the YOLOv8 predictor
using var predictor = new YoloPredictor("path/to/model.onnx");

// Run model
var result = predictor.Detect("path/to/image.jpg");
// or
var result = await predictor.DetectAsync("path/to/image.jpg");

// Write result summary to terminal
Console.WriteLine(result);
```
# Plotting

You can to plot the target image for preview the model results, this code demonstrates how to run a inference, plot the results on image and save to file:

```csharp
using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;
using SixLabors.ImageSharp;

// Load the YOLOv8 predictor
using var predictor = new YoloPredictor("path/to/model.onnx");

// Load the target image
using var image = Image.Load("path/to/image");

// Run model
var result = await predictor.PoseAsync(image);

// Create plotted image from model results
using var plotted = await result.PlotImageAsync(image);

// Write the plotted image to file
plotted.Save("./pose_demo.jpg")
```

You can also predict and save to file in one operation:

```csharp
using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;
using SixLabors.ImageSharp;

// Load the YOLOv8 predictor
using var predictor = new YoloPredictor("path/to/model.onnx");

// Run model, plot predictions and write to file
predictor.PredictAndSaveAsync("path/to/image");
```
## Example Images:

#### Detection:

![detect-demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/Assets/detect-demo.jpg)

#### Pose:

![pose-demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/Assets/pose-demo.jpg)

#### Segmentation:

![seg-demo!](https://raw.githubusercontent.com/dme-compunet/YOLOv8/main/Assets/seg-demo.jpg)

# License

AGPL-3.0 License

**Important Note:** This project depends on ImageSharp, you should check the license details [here](https://github.com/SixLabors/ImageSharp/blob/main/LICENSE)