namespace Compunet.YoloV8.Services.Plotting;

internal class BoxDrawer : IBoxDrawer
{
    private const float BoxFillAlpha = .1f;

    public void DrawBox(YoloPrediction prediction, PlottingContext context)
    {
        PointF[] points = prediction switch
        {
            ObbDetection obb => GetBoxPoints(obb),
            Detection detection => GetBoxPoints(detection),
            _ => throw new InvalidOperationException("The prediction is not contains a bounding box"),
        };

        var polygon = new Polygon(points);
        var color = context.ColorPalette.GetColor(prediction.Name.Id);

        context.Target.Mutate(x =>
        {
            // Fill the box
            x.Fill(color.WithAlpha(BoxFillAlpha), polygon);

            // Draw the box bounds
            x.Draw(color, context.BorderThickness, polygon);
        });
    }

    private static PointF[] GetBoxPoints(Detection detection)
    {
        var rectangle = detection.Bounds;

        return
        [
            new PointF(rectangle.Left, rectangle.Top),
            new PointF(rectangle.Right, rectangle.Top),
            new PointF(rectangle.Right, rectangle.Bottom),
            new PointF(rectangle.Left, rectangle.Bottom),
        ];
    }

    private static PointF[] GetBoxPoints(ObbDetection detection)
    {
        var points = detection.GetCornerPoints();

        return [.. points.Select(point => new PointF(point.X, point.Y))];
    }
}