namespace Compunet.YoloV8.Data;

public abstract class YoloV8Result
{
    public required Size Image { get; init; }

    public required SpeedResult Speed { get; init; }
}