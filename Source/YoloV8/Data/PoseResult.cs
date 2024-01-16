namespace Compunet.YoloV8.Data;

public class PoseResult : YoloV8Result
{
    public required IEnumerable<PoseBoundingBox> Boxes { get; init; }
}