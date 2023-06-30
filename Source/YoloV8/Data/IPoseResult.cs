namespace Compunet.YoloV8.Data;

public interface IPoseResult : IYoloV8Result
{
    IReadOnlyList<IPoseBoundingBox> Boxes { get; }
}