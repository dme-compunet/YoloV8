namespace Compunet.YoloV8.Data;

public interface IBoundingBox
{
    YoloV8Class Class { get; }

    public Rectangle Bounds { get; }

    public float Confidence { get; }
}