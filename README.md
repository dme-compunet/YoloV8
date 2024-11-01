# YoloSharp

🎥 Use [YOLO11](https://github.com/ultralytics/ultralytics) in real-time for object detection tasks 🚀 with edge performance ⚡️ powered by ONNX-Runtime.

# Features

- **YOLO Tasks** 🌟 Support for all YOLO vision tasks ([Detect](https://docs.ultralytics.com/tasks/detect) | [OBB](https://docs.ultralytics.com/tasks/obb) | [Pose](https://docs.ultralytics.com/tasks/pose) | [Segment](https://docs.ultralytics.com/tasks/segment) | [Classify](https://docs.ultralytics.com/tasks/classify))
- **High Performance** 🚀 Various techniques and use of .NET features to maximize performance
- **Reduced Memory Usage** 🧠 By reusing memory blocks and reducing the pressure on the GC
- **Plotting Options** ✏️ Draw the predictions on the target image to preview the model results
- **YOLO Versions** 🔧 Includes support for: [YOLOv8](https://docs.ultralytics.com/models/yolov8) [YOLOv10](https://docs.ultralytics.com/models/yolov10) [YOLO11](https://docs.ultralytics.com/models/yolo11)

# Installation

This project provides two NuGet packages:

- For CPU inference, use the package: [YoloSharp](https://www.nuget.org/packages/YoloSharp) (includes the [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) package)
- For GPU inference, use the package: [YoloSharp.Gpu](https://www.nuget.org/packages/YoloSharp.Gpu) (includes the [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu) package)

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

Add the `YoloSharp` (or `YoloSharp.Gpu`) package to your project:

```shell
dotnet add package YoloSharp
```

Use the following C# code to load the model and run basic prediction:

```csharp
using Compunet.YoloSharp;

// Load the YOLO predictor
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
using Compunet.YoloSharp;
using Compunet.YoloSharp.Plotting;
using SixLabors.ImageSharp;

// Load the YOLO predictor
using var predictor = new YoloPredictor("path/to/model.onnx");

// Load the target image
using var image = Image.Load("path/to/image");

// Run model
var result = await predictor.PoseAsync(image);

// Create plotted image from model results
using var plotted = await result.PlotImageAsync(image);

// Write the plotted image to file
plotted.Save("./pose_demo.jpg");
```

You can also predict and save to file in one operation:

```csharp
using Compunet.YoloSharp;
using Compunet.YoloSharp.Plotting;
using SixLabors.ImageSharp;

// Load the YOLO predictor
using var predictor = new YoloPredictor("path/to/model.onnx");

// Run model, plot predictions and write to file
predictor.PredictAndSaveAsync("path/to/image");
```

## Example Images:

#### Detection:

![detect-demo!](https://raw.githubusercontent.com/dme-compunet/YoloSharp/main/Assets/detect-demo.jpg)

#### Pose:

![pose-demo!](https://raw.githubusercontent.com/dme-compunet/YoloSharp/main/Assets/pose-demo.jpg)

#### Segmentation:

![seg-demo!](https://raw.githubusercontent.com/dme-compunet/YoloSharp/main/Assets/seg-demo.jpg)

# License

AGPL-3.0 License

**Important Note:** This project depends on ImageSharp, you should check the license details [here](https://github.com/SixLabors/ImageSharp/blob/main/LICENSE)
