using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;
using SixLabors.ImageSharp;

var output = "./assets/output";

if (Directory.Exists(output) == false)
    Directory.CreateDirectory(output);

YoloV8Parameters.Default.ProcessWithOriginalAspectRatio = true;

PoseDemo("./assets/input/sports.jpg", "./assets/models/yolov8s-pose.onnx");
DetectDemo("./assets/input/bus.jpg", "./assets/models/yolov8s.onnx");
SegmentDemo("./assets/input/sports.jpg", "./assets/models/yolov8s-seg.onnx");
ClassifyDemo(new string[]
{
    "./assets/input/pizza.jpg",
    "./assets/input/teddy.jpg",
    "./assets/input/toaster.jpg",
}, "./assets/models/yolov8s-cls.onnx");

void PoseDemo(string image, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ POSE DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    Console.WriteLine("Working...");
    var result = predictor.Pose(image);

    Console.WriteLine();

    Console.WriteLine($"Result: {result}");
    Console.WriteLine($"Speed: {result.Speed}");

    Console.WriteLine();

    Console.WriteLine("Plotting and saving...");
    using var origin = Image.Load(image);

    using var ploted = result.PlotImage(origin);

    var pathToSave = Path.Combine(output, Path.GetFileName(image));

    ploted.Save(pathToSave);
}

void DetectDemo(string image, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ DETECTION DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    Console.WriteLine("Working...");
    var result = predictor.Detect(image);

    Console.WriteLine();

    Console.WriteLine($"Result: {result}");
    Console.WriteLine($"Speed: {result.Speed}");

    Console.WriteLine();

    Console.WriteLine("Plotting and saving...");
    using var origin = Image.Load(image);

    using var ploted = result.PlotImage(origin);

    var pathToSave = Path.Combine(output, Path.GetFileName(image));

    ploted.Save(pathToSave);
}

void SegmentDemo(string image, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ SEGMENTATION DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    Console.WriteLine("Working...");
    var result = predictor.Segment(image);

    Console.WriteLine();

    Console.WriteLine($"Result: {result}");
    Console.WriteLine($"Speed: {result.Speed}");

    Console.WriteLine();

    Console.WriteLine("Plotting and saving...");
    using var origin = Image.Load(image);

    using var ploted = result.PlotImage(origin, new SegmentationPlottingOptions { MaskConfidence = .65F });

    var filename = $"{Path.GetFileNameWithoutExtension(image)}_seg";
    var extension = Path.GetExtension(image);

    var pathToSave = Path.Combine(output, filename + extension);

    ploted.Save(pathToSave);
}

void ClassifyDemo(string[] images, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ POSE DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    foreach (var image in images)
    {
        Console.WriteLine("Working... ({0})", image);
        var result = predictor.Classify(image);

        Console.WriteLine();

        Console.WriteLine($"Result: {result}");
        Console.WriteLine($"Speed: {result.Speed}");

        Console.WriteLine();

        Console.WriteLine("Plotting and saving...");
        using var origin = Image.Load(image);

        using var ploted = result.PlotImage(origin);

        var pathToSave = Path.Combine(output, Path.GetFileName(image));

        ploted.Save(pathToSave);

        Console.WriteLine();
    }
}