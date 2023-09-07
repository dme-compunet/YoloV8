namespace Compunet.YoloV8.Data;

internal abstract class YoloV8Result : IYoloV8Result
{
    public Size Image { get; }

    public SpeedResult Speed { get; }

    public YoloV8Result(Size image, SpeedResult speed)
    {
        Image = image;
        Speed = speed;
    }
}