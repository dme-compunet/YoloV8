namespace Compunet.YoloV8.Data;

public class BoundingBox
{
    public required YoloV8Class Class { get; init; }

    public required Rectangle Bounds { get; init; }

    public required float Confidence { get; init; }

    public override string ToString()
    {
        return $"{Class.Name} ({Confidence:N})";
    }
}