namespace Compunet.YoloV8.Data;

internal class BoundingBox(YoloV8Class name, Rectangle bounds, float confidence) : IBoundingBox
{
    public YoloV8Class Class { get; } = name;

    public Rectangle Bounds { get; } = bounds;

    public float Confidence { get; } = confidence;

    public override string ToString()
    {
        return $"{Class.Name} ({Confidence:N})";
    }
}