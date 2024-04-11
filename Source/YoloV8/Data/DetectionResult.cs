namespace Compunet.YoloV8.Data;

public class DetectionResult : YoloV8Result
{
    public required BoundingBox[] Boxes { get; init; }

    public override string ToString() => Boxes.Summary();
}