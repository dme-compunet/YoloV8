namespace Compunet.YoloV8.Data;

public class SegmentationResult : YoloV8Result
{
    public required SegmentationBoundingBox[] Boxes { get; init; }
}