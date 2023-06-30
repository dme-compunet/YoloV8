using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Data;

internal class BoundingBox : IBoundingBox
{
    public YoloV8Class Class { get; }

    public Rectangle Rectangle { get; }

    public float Confidence { get; }

    public BoundingBox(YoloV8Class name, Rectangle rectangle, float confidence)
    {
        Class = name;
        Rectangle = rectangle;
        Confidence = confidence;
    }

    public override string ToString()
    {
        return $"{Class.Name} ({Confidence:N})";
    }
}