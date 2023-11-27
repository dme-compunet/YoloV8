namespace Compunet.YoloV8.Parsers;

internal readonly struct IndexedBoundingBox(int index, YoloV8Class name, Rectangle bounds, float confidence)
{
    internal bool IsEmpty => Bounds.IsEmpty;

    public int Index { get; } = index;

    public YoloV8Class Class { get; } = name;

    public Rectangle Bounds { get; } = bounds;

    public float Confidence { get; } = confidence;
}