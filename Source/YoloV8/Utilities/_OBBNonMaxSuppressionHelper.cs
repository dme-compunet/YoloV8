namespace Compunet.YoloV8.Utilities;

internal static class ObbNonMaxSuppressionHelper
{
    public static ObbIndexedBoundingBox[] Suppress(ObbIndexedBoundingBox[] boxes, float iouThreshold)
    {
        Array.Sort(boxes);

        var boxCount = boxes.Length;

        var activeCount = boxCount;

        var isNotActiveBoxes = new bool[boxCount];

        var selected = new List<ObbIndexedBoundingBox>();

        for (int i = 0; i < boxCount; i++)
        {
            if (isNotActiveBoxes[i])
            {
                continue;
            }

            var boxA = boxes[i];

            selected.Add(boxA);

            for (var j = i + 1; j < boxCount; j++)
            {
                if (isNotActiveBoxes[j])
                {
                    continue;
                }

                var boxB = boxes[j];

                if (CalculateIoU(boxA, boxB) > iouThreshold)
                {
                    isNotActiveBoxes[j] = true;

                    activeCount--;

                    if (activeCount <= 0)
                    {
                        break;
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

    private static double CalculateIoU(this ObbIndexedBoundingBox box1, ObbIndexedBoundingBox box2)
    {
        var areaA = Area(box1);

        if (areaA <= 0f)
        {
            return 0f;
        }

        var areaB = Area(box2);

        if (areaB <= 0f)
        {
            return 0f;
        }

        var vertices1 = box1.GetCornerPoints();
        var vertices2 = box2.GetCornerPoints();

        var rect1 = new Path64(vertices1.Select(v => new Point64(v.X, v.Y)));
        var rect2 = new Path64(vertices2.Select(v => new Point64(v.X, v.Y)));

        var subject = new Paths64([rect1]);
        var clip = new Paths64([rect2]);

        var intersection = Clipper.Intersect(subject, clip, FillRule.EvenOdd);
        var union = Clipper.Union(subject, clip, FillRule.EvenOdd);

        if (intersection.Count == 0 || union.Count == 0)
        {
            return 0f;
        }

        var intersectionArea = Clipper.Area(intersection[0]);
        var unionArea = Clipper.Area(union[0]);

        return intersectionArea / unionArea;
    }

    private static int Area(ObbIndexedBoundingBox obb)
    {
        return obb.Bounds.Width * obb.Bounds.Height;
    }
}