namespace Compunet.YoloV8.Parsers;

internal readonly struct IndexedBoundingBox : IComparable<IndexedBoundingBox>
{
    public bool IsEmpty => Bounds.IsEmpty;

    public required int Index { get; init; }

    public required YoloV8Class Class { get; init; }

    public required Rectangle Bounds { get; init; }

    public required float Confidence { get; init; }

    public int CompareTo(IndexedBoundingBox other) => Confidence.CompareTo(other.Confidence);
}