namespace Compunet.YoloV8.Parsing;

internal struct RawBoundingBox : IRawBoundingBox<RawBoundingBox>
{
    public required int Index { get; init; }

    public required int NameIndex { get; init; }

    public required RectangleF Bounds { get; set; }

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

        var intersection = RectangleF.Intersect(rect1, rect2);
        var intersectionArea = intersection.Width * intersection.Height;

        return (float)intersectionArea / (area1 + area2 - intersectionArea);
    }

    public static RawBoundingBox Parse(ref RawParsingContext context, int index, int nameIndex, float confidence)
    {
        var tensor = context.Tensor;
        var tensorSpan = tensor.Span;
        var stride1 = tensor.Strides[1];

        RectangleF bounds;

        if (context.Architecture == YoloArchitecture.YoloV10)
        {
            var boxOffset = index * stride1;

            var xMin = (int)tensorSpan[boxOffset + 0];
            var yMin = (int)tensorSpan[boxOffset + 1];
            var xMax = (int)tensorSpan[boxOffset + 2];
            var yMax = (int)tensorSpan[boxOffset + 3];

            bounds = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin);
        }
        else // YOLOv8
        {
            var x = tensorSpan[0 + index];
            var y = tensorSpan[1 * stride1 + index];
            var w = tensorSpan[2 * stride1 + index];
            var h = tensorSpan[3 * stride1 + index];

            bounds = new RectangleF(x - w / 2, y - h / 2, w, h);
        }

        return new RawBoundingBox
        {
            Index = index,
            Bounds = bounds,
            NameIndex = nameIndex,
            Confidence = confidence,
        };
    }

    public readonly int CompareTo(RawBoundingBox other) => Confidence.CompareTo(other.Confidence);
}