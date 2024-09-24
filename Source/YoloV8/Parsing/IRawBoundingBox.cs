namespace Compunet.YoloV8.Parsing;

internal interface IRawBoundingBox<TSelf> : IComparable<TSelf>
{
    public bool IsEmpty => Bounds.IsEmpty;

    public YoloName Name { get; }

    public Rectangle Bounds { get; }

    public float Confidence { get; }

    public static abstract float CalculateIoU(ref TSelf box1, ref TSelf box2);

    public static abstract TSelf Parse(ref RawParsingContext context, int index, YoloName name, float confidence, YoloArchitecture architecture);
}