using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Data;

internal class BoundingBox : IBoundingBox
{
    public YoloV8Class Name { get; }

    public Rectangle Rectangle { get; }

    public float Confidence { get; }

    public BoundingBox(YoloV8Class name, Rectangle rectangle, float confidence)
    {
        Name = name;
        Rectangle = rectangle;
        Confidence = confidence;
    }

    public override string ToString()
    {
        return $"{Name.Name} ({Confidence:N})";
    }
}