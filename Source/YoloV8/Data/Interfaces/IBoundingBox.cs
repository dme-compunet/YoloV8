using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Data;

public interface IBoundingBox
{
    YoloV8Class Class { get; }

    public Rectangle Rectangle { get; }

    public float Confidence { get; }
}