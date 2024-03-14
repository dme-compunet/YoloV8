using ClipperLib;

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

    private static double CalculateIoU(this ObbIndexedBoundingBox boxA, ObbIndexedBoundingBox boxB)
    {
        var areaA = Area(boxA);

        if (areaA <= 0f)
        {
            return 0f;
        }

        var areaB = Area(boxB);

        if (areaB <= 0f)
        {
            return 0f;
        }

        var vertices1 = boxA.GetCornerPoints();
        var vertices2 = boxB.GetCornerPoints();

        var rect1 = vertices1.Select(v => new IntPoint(v.X, v.Y)).ToList();
        var rect2 = vertices2.Select(v => new IntPoint(v.X, v.Y)).ToList();

        var clipper = new Clipper();

        var intersection = new List<List<IntPoint>>();
        var union = new List<List<IntPoint>>();

        clipper.AddPath(rect1, PolyType.ptSubject, true);
        clipper.AddPath(rect2, PolyType.ptClip, true);

        clipper.Execute(ClipType.ctIntersection, intersection, PolyFillType.pftEvenOdd, PolyFillType.pftEvenOdd);
        clipper.Execute(ClipType.ctUnion, union, PolyFillType.pftEvenOdd, PolyFillType.pftEvenOdd);

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