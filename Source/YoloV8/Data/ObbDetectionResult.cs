namespace Compunet.YoloV8.Data;

public class ObbDetectionResult : YoloV8Result
{
    public required ObbBoundingBox[] Boxes { get; init; }

    public override string ToString() => Boxes.Summary();
}