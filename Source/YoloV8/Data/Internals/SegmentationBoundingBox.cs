using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Data;

internal class SegmentationBoundingBox : BoundingBox, ISegmentationBoundingBox
{
    public float[,] Mask { get; }

    public SegmentationBoundingBox(YoloV8Class name,
                                   Rectangle rectangle,
                                   float confidence,
                                   float[,] mask) : base(name, rectangle, confidence)
    {
        Mask = mask;
    }
}