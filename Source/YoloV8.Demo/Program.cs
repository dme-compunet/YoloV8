using Compunet.YoloV8;
using System.Diagnostics;

Console.WriteLine("Loading pose estimation model...");
using var posePredictor = new YoloPredictor("./models/yolov8n-pose-uint8.onnx");

Console.WriteLine("Loading detection model...");
using var detectPredictor = new YoloPredictor("./models/yolov8n-uint8.onnx");

Console.WriteLine("Loading segmentation model...");
using var segmentPredictor = new YoloPredictor("./models/yolov8n-seg-uint8.onnx");

Console.WriteLine("Loading classification model...");
using var classifyPredictor = new YoloPredictor("./models/yolov8n-cls-uint8.onnx");

Console.WriteLine();

await PredictAndSaveAsync(posePredictor, "bus.jpg");
await PredictAndSaveAsync(posePredictor, "sports.jpg");

await PredictAndSaveAsync(detectPredictor, "bus.jpg");
await PredictAndSaveAsync(detectPredictor, "sports.jpg");

await PredictAndSaveAsync(segmentPredictor, "sports.jpg");

await PredictAndSaveAsync(classifyPredictor, "pizza.jpg");
await PredictAndSaveAsync(classifyPredictor, "teddy.jpg");
await PredictAndSaveAsync(classifyPredictor, "toaster.jpg");

if (OperatingSystem.IsWindows())
{
    Process.Start(new ProcessStartInfo
    {
        FileName = Path.GetFullPath("./images"),
        UseShellExecute = true,
    });
}

static async Task PredictAndSaveAsync(YoloPredictor predictor, string image)
{
    var path = $"./images/{image}";
    var task = predictor.Metadata.Task;

    Console.WriteLine($"Running '{image}' (test: {task})...");

    var result = await predictor.PredictAndSaveAsync(path);

    Console.WriteLine($"Result: {result}");
    Console.WriteLine($"Speed:  {result.Speed}");

    Console.WriteLine();
}