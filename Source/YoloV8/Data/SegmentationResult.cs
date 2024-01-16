namespace Compunet.YoloV8.Data;

public class SegmentationResult : YoloV8Result
{
    public required IEnumerable<SegmentationBoundingBox> Boxes { get; init; }
}