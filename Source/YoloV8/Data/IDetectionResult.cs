namespace Compunet.YoloV8.Data;

public interface IDetectionResult : IYoloV8Result
{
    IReadOnlyList<IBoundingBox> Boxes { get; }
}