namespace Compunet.YoloV8.Data;

public class DetectionResult : YoloV8Result
{
    public required IEnumerable<BoundingBox> Boxes { get; init; }

    public override string ToString()
    {
        return Boxes.Summary();
    }
}