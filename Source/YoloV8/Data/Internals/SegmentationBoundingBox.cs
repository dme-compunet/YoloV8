namespace Compunet.YoloV8.Data;

internal class SegmentationBoundingBox(YoloV8Class name,
                                       Rectangle bounds,
                                       float confidence,
                                       IMask mask) : BoundingBox(name, bounds, confidence), ISegmentationBoundingBox
{
    public IMask Mask { get; } = mask;
}