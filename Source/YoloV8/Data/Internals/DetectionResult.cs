namespace Compunet.YoloV8.Data;

internal class DetectionResult(Size image,
                               SpeedResult speed,
                               IReadOnlyList<IBoundingBox> boxes) : YoloV8Result(image, speed), IDetectionResult
{
    public IReadOnlyList<IBoundingBox> Boxes { get; } = boxes;

    public override string ToString()
    {
        return Boxes.Summary();
    }
}