namespace Compunet.YoloV8.Services.Plotting;

internal class ObbDetectionPlotter(IBoxDrawer boxPlotter,
                                   INameDrawer namePlotter) : IPlotter<ObbDetection>
{
    public void Plot(YoloResult<ObbDetection> result, PlottingContext context)
    {
        foreach (var box in result)
        {
            var points = GetBoxPoints(box);

            boxPlotter.DrawBox(box, points, context);
            namePlotter.DrawName(box, points[0], false, context);
        }
    }

    private static PointF[] GetBoxPoints(ObbDetection detection)
    {
        var points = detection.GetCornerPoints();

        return [.. points.Select(point => new PointF(point.X, point.Y))];
    }
}