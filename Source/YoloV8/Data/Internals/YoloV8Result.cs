namespace Compunet.YoloV8.Data;

internal abstract class YoloV8Result(Size image, SpeedResult speed) : IYoloV8Result
{
    public Size Image { get; } = image;

    public SpeedResult Speed { get; } = speed;
}