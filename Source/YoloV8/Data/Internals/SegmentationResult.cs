namespace Compunet.YoloV8.Data;

internal class SegmentationResult(Size image,
                                  SpeedResult speed,
                                  IReadOnlyList<ISegmentationBoundingBox> boxes) : YoloV8Result(image, speed), ISegmentationResult
{
    public IReadOnlyList<ISegmentationBoundingBox> Boxes { get; } = boxes;

    public override string ToString()
    {
        return Boxes.Summary();
    }
}