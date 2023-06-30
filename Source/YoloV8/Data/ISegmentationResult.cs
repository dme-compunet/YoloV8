namespace Compunet.YoloV8.Data;

public interface ISegmentationResult : IYoloV8Result
{
    IReadOnlyList<ISegmentationBoundingBox> Boxes { get; }
}