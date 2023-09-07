namespace Compunet.YoloV8.Data;

internal class BoundingBox : IBoundingBox
{
    public YoloV8Class Class { get; }

    public Rectangle Bounds { get; }

    public float Confidence { get; }

    public BoundingBox(YoloV8Class name, Rectangle bounds, float confidence)
    {
        Class = name;
        Bounds = bounds;
        Confidence = confidence;
    }

    public override string ToString()
    {
        return $"{Class.Name} ({Confidence:N})";
    }
}