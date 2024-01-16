namespace Compunet.YoloV8.Data;

public class SegmentationBoundingBox : BoundingBox
{
    public required SegmentationMask Mask { get; init; }
}