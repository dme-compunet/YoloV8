namespace Compunet.YoloV8.Data;

internal class SegmentationResult : YoloV8Result, ISegmentationResult
{
    public IReadOnlyList<ISegmentationBoundingBox> Boxes { get; }

    public SegmentationResult(Size image,
                              SpeedResult speed,
                              IReadOnlyList<ISegmentationBoundingBox> boxes)
        : base(image, speed)
    {
        Boxes = boxes;
    }

    public override string ToString()
    {
        return Boxes.Summary();
    }
}