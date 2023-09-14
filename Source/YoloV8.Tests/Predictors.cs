namespace YoloV8.Tests;

public static class Predictors
{
    static Predictors()
    {
        YoloV8Parameters.Default.ProcessWithOriginalAspectRatio = true;
    }

    public static readonly YoloV8Predictor Pose = new("./assets/models/yolov8s-pose.onnx");
    public static readonly YoloV8Predictor Detection = new("./assets/models/yolov8s.onnx");
    public static readonly YoloV8Predictor Segmentation = new("./assets/models/yolov8s-seg.onnx");
    public static readonly YoloV8Predictor Classification = new("./assets/models/yolov8s-cls.onnx");

    public static YoloV8Predictor GetPredictor(YoloV8Task task)
    {
        return task switch
        {
            YoloV8Task.Pose => Pose,
            YoloV8Task.Detect => Detection,
            YoloV8Task.Segment => Segmentation,
            YoloV8Task.Classify => Classification,
            _ => throw new InvalidEnumArgumentException()
        };
    }
}