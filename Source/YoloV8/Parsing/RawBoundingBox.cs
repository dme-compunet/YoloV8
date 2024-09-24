namespace Compunet.YoloV8.Parsing;

internal readonly struct RawBoundingBox : IRawBoundingBox<RawBoundingBox>
{
    public required int Index { get; init; }

    public required YoloName Name { get; init; }

    public required Rectangle Bounds { get; init; }

    public required float Confidence { get; init; }

    public static float CalculateIoU(ref RawBoundingBox box1, ref RawBoundingBox box2)
    {
        var rect1 = box1.Bounds;
        var rect2 = box2.Bounds;

        var area1 = rect1.Width * rect1.Height;

        if (area1 <= 0f)
        {
            return 0f;
        }

        var area2 = rect2.Width * rect2.Height;

        if (area2 <= 0f)
        {
            return 0f;
        }

        var intersection = Rectangle.Intersect(rect1, rect2);
        var intersectionArea = intersection.Width * intersection.Height;

        return (float)intersectionArea / (area1 + area2 - intersectionArea);
    }

    public static RawBoundingBox Parse(ref RawParsingContext context, int index, YoloName name, float confidence, YoloArchitecture architecture)
    {
        var tensor = context.Tensor;
        var tensorSpan = tensor.Buffer.Span;
        var stride1 = context.Stride1;
        var padding = context.Padding;
        var ratio = context.Ratio;

        int xMin;
        int yMin;
        int xMax;
        int yMax;

        if (architecture == YoloArchitecture.YoloV10)
        {
            var boxOffset = index * stride1;

            var x = tensorSpan[boxOffset + 0];
            var y = tensorSpan[boxOffset + 1];
            var w = tensorSpan[boxOffset + 2];
            var h = tensorSpan[boxOffset + 3];

            xMin = (int)((x - padding.X) * ratio.X);
            yMin = (int)((y - padding.Y) * ratio.Y);
            xMax = (int)((w - padding.X) * ratio.X);
            yMax = (int)((h - padding.Y) * ratio.X);
        }
        else // YOLOv8
        {
            var x = tensorSpan[0 + index];
            var y = tensorSpan[1 * stride1 + index];
            var w = tensorSpan[2 * stride1 + index];
            var h = tensorSpan[3 * stride1 + index];

            xMin = (int)((x - w / 2 - padding.X) * ratio.X);
            yMin = (int)((y - h / 2 - padding.Y) * ratio.Y);
            xMax = (int)((x + w / 2 - padding.X) * ratio.X);
            yMax = (int)((y + h / 2 - padding.Y) * ratio.Y);
        }

        var bounds = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);

        return new RawBoundingBox
        {
            Index = index,
            Bounds = bounds,
            Name = name,
            Confidence = confidence,
        };
    }

    public int CompareTo(RawBoundingBox other) => Confidence.CompareTo(other.Confidence);
}