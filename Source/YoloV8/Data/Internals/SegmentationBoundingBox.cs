namespace Compunet.YoloV8.Data;

internal class SegmentationBoundingBox : BoundingBox, ISegmentationBoundingBox
{
    public IMask Mask { get; }

    public SegmentationBoundingBox(YoloV8Class name,
                                   Rectangle bounds,
                                   float confidence,
                                   IMask mask) : base(name, bounds, confidence)
    {
        Mask = mask;
    }
}