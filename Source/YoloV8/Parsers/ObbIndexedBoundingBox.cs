namespace Compunet.YoloV8.Parsers;

internal readonly struct ObbIndexedBoundingBox : IComparable<ObbIndexedBoundingBox>
{
    public bool IsEmpty => Bounds == default;

    public required int Index { get; init; }

    public required YoloV8Class Class { get; init; }

    public required Rectangle Bounds { get; init; }

    public required float Angle { get; init; }

    public required float Confidence { get; init; }

    public int CompareTo(ObbIndexedBoundingBox other) => Confidence.CompareTo(other.Confidence);
}