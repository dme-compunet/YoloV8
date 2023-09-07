namespace Compunet.YoloV8.Data;

internal class DetectionResult : YoloV8Result, IDetectionResult
{
    public IReadOnlyList<IBoundingBox> Boxes { get; }

    public DetectionResult(Size image,
                           SpeedResult speed,
                           IReadOnlyList<IBoundingBox> boxes)
        : base(image, speed)
    {
        Boxes = boxes;
    }

    public override string ToString()
    {
        return Boxes.Summary();
    }
}