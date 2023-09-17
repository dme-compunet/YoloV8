# YOLOv8

Use [YOLOv8](https://github.com/ultralytics/ultralytics) in real-time, for object detection, instance segmentation, pose estimation and image classification, via [ONNX Runtime](https://github.com/microsoft/onnxruntime)

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

You can plot the model prediction results for preview, use in following code to predict and plot a image and save to file:

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
