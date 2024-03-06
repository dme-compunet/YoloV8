namespace Compunet.YoloV8.Utilities;

internal static class NonMaxSuppressionHelper
{
    public static IndexedBoundingBox[] Suppress(IndexedBoundingBox[] boxes, float iouThreshold)
    {
        Array.Sort(boxes);

        var boxCount = boxes.Length;

        var activeCount = boxCount;

        var isNotActiveBoxes = new bool[boxCount];

        var selected = new List<IndexedBoundingBox>();

        for (int i = boxCount - 1; i >= 0; i--)
        {
            if (isNotActiveBoxes[i])
            {
                continue;
            }

            var boxA = boxes[i];

            selected.Add(boxA);

            for (var j = i; j >= 0; j--)
            {
                if (isNotActiveBoxes[j])
                {
                    continue;
                }

                var boxB = boxes[j];

                if (boxA.Class == boxB.Class)
                {
                    if (CalculateIoU(boxA.Bounds, boxB.Bounds) > iouThreshold)
                    {
                        isNotActiveBoxes[j] = true;

                        activeCount--;

                        if (activeCount <= 0)
                        {
                            break;
                        }
                    }
                }
            }

            if (activeCount <= 0)
            {
                break;
            }
        }

        return [.. selected];
    }

    private static float CalculateIoU(Rectangle rectA, Rectangle rectB)
    {
        var areaA = Area(rectA);

        if (areaA <= 0f)
        {
            return 0f;
        }

        var areaB = Area(rectB);

        if (areaB <= 0f)
        {
            return 0f;
        }

        var intersectionArea = Area(Rectangle.Intersect(rectA, rectB));

        return (float)intersectionArea / (areaA + areaB - intersectionArea);
    }

    private static int Area(Rectangle rectangle)
    {
        return rectangle.Width * rectangle.Height;
    }
}