using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;
using SixLabors.ImageSharp;

var output = "./assets/output";

if (Directory.Exists(output) == false)
    Directory.CreateDirectory(output);

await PoseDemo("./assets/input/sports.jpg", "./assets/models/yolov8s-pose.onnx");

await DetectDemo("./assets/input/bus.jpg", "./assets/models/yolov8s.onnx");

await SegmentDemo("./assets/input/sports.jpg", "./assets/models/yolov8s-seg.onnx");

await ClassifyDemo(new string[]
{
    "./assets/input/pizza.jpg",
    "./assets/input/teddy.jpg",
    "./assets/input/toaster.jpg",
}, "./assets/models/yolov8s-cls.onnx");

async Task PoseDemo(string image, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ POSE DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    Console.WriteLine("Working...");
    var result = await predictor.PoseAsync(image);

    Console.WriteLine();

    Console.WriteLine($"Result: {result}");
    Console.WriteLine($"Speed: {result.Speed}");

    Console.WriteLine();

    Console.WriteLine("Plotting and saving...");
    using var origin = Image.Load(image);

    using var ploted = await result.PlotImageAsync(origin);

    var pathToSave = Path.Combine(output, Path.GetFileName(image));

    ploted.Save(pathToSave);
}

async Task DetectDemo(string image, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ DETECTION DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    Console.WriteLine("Working...");
    var result = await predictor.DetectAsync(image);

    Console.WriteLine();

    Console.WriteLine($"Result: {result}");
    Console.WriteLine($"Speed: {result.Speed}");

    Console.WriteLine();

    Console.WriteLine("Plotting and saving...");
    using var origin = Image.Load(image);

    using var ploted = await result.PlotImageAsync(origin);

    var pathToSave = Path.Combine(output, Path.GetFileName(image));

    ploted.Save(pathToSave);
}

async Task SegmentDemo(string image, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ SEGMENTATION DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    Console.WriteLine("Working...");
    var result = await predictor.SegmentAsync(image);

    Console.WriteLine();

    Console.WriteLine($"Result: {result}");
    Console.WriteLine($"Speed: {result.Speed}");

    Console.WriteLine();

    Console.WriteLine("Plotting and saving...");
    using var origin = Image.Load(image);

    using var ploted = await result.PlotImageAsync(origin, new SegmentationPlottingOptions { MaskConfidence = .65F });

    var filename = $"{Path.GetFileNameWithoutExtension(image)}_seg";
    var extension = Path.GetExtension(image);

    var pathToSave = Path.Combine(output, filename + extension);

    ploted.Save(pathToSave);
}

async Task ClassifyDemo(string[] images, string model)
{
    Console.WriteLine();
    Console.WriteLine("================ POSE DEMO ================");
    Console.WriteLine();

    Console.WriteLine("Loading model...");
    using var predictor = new YoloV8(model);

    foreach (var image in images)
    {
        Console.WriteLine("Working... ({0})", image);
        var result = await predictor.ClassifyAsync(image);

        Console.WriteLine();

        Console.WriteLine($"Result: {result}");
        Console.WriteLine($"Speed: {result.Speed}");

        Console.WriteLine();

        Console.WriteLine("Plotting and saving...");
        using var origin = Image.Load(image);

        using var ploted = await result.PlotImageAsync(origin);

        var pathToSave = Path.Combine(output, Path.GetFileName(image));

        ploted.Save(pathToSave);

        Console.WriteLine();
    }
}