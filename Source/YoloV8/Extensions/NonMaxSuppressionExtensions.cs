namespace Compunet.YoloV8.Extensions;

internal static class NonMaxSuppressionExtensions
{
    public static IReadOnlyList<IndexedBoundingBox> NonMaxSuppression(this IReadOnlyList<IndexedBoundingBox> boxes, float threshold)
    {
        var sorted = boxes.OrderByDescending(x => x.Confidence).ToArray();
        var count = sorted.Length;

        var activeCount = count;
        var isActiveBoxes = new bool[count];

        Array.Fill(isActiveBoxes, true);

        var selected = new List<IndexedBoundingBox>();

        for (int i = 0; i < count; i++)
        {
            if (isActiveBoxes[i])
            {
                var boxA = sorted[i];

                selected.Add(boxA);

                for (var j = i + 1; j < count; j++)
                {
                    if (isActiveBoxes[j])
                    {
                        var boxB = sorted[j];

                        if (CalculateIoU(boxA.Bounds, boxB.Bounds) > threshold)
                        {
                            isActiveBoxes[j] = false;
                            activeCount--;

                            if (activeCount <= 0)
                                break;
                        }
                    }
                }

                if (activeCount <= 0)
                    break;
            }
        }

        return selected;
    }

    private static float CalculateIoU(Rectangle first, Rectangle second)
    {
        var areaA = Area(first);

        if (areaA <= 0f)
            return 0f;

        var areaB = Area(second);

        if (areaB <= 0f)
            return 0f;

        var intersection = Rectangle.Intersect(first, second);
        var intersectionArea = Area(intersection);

        return (float)intersectionArea / (areaA + areaB - intersectionArea);
    }

    private static int Area(Rectangle rectangle)
    {
        return rectangle.Width * rectangle.Height;
    }
}