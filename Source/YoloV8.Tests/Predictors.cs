namespace YoloV8.Tests;

public static class Predictors
{
    public static YoloPredictor Pose { get; } = new("./models/yolov8n-pose-uint8.onnx");
    public static YoloPredictor Detection { get; } = new("./models/yolov8n-uint8.onnx");
    public static YoloPredictor Segmentation { get; } = new("./models/yolov8n-seg-uint8.onnx");
    public static YoloPredictor Classification { get; } = new("./models/yolov8n-cls-uint8.onnx");

    public static YoloPredictor GetPredictor(YoloTask task)
    {
        return task switch
        {
            YoloTask.Pose => Pose,
            YoloTask.Detect => Detection,
            YoloTask.Segment => Segmentation,
            YoloTask.Classify => Classification,
            _ => throw new InvalidEnumArgumentException()
        };
    }
}