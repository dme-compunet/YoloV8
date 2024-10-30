namespace Compunet.YoloSharp.Parsing;

internal interface IRawBoundingBox<TSelf> : IComparable<TSelf>
{
    public int NameIndex { get; }

    public RectangleF Bounds { get; }

    public float Confidence { get; }

    public static abstract float CalculateIoU(ref TSelf box1, ref TSelf box2);

    public static abstract TSelf Parse(ref RawParsingContext context, int index, int nameIndex, float confidence);
}