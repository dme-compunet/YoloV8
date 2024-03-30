namespace Compunet.YoloV8.Plotting;

public static class PlottingExtensions
{
    public static Image PlotImage(this PoseResult result, ImageSelector<Rgba32> originImage, PosePlottingOptions? options = null)
    {
        options ??= PosePlottingOptions.Default;

        var process = originImage.Load(true);

        EnsureSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var boxBorderThickness = options.BoxBorderThickness * ratio;

        var radius = options.KeypointRadius * ratio;
        var lineThickness = options.KeypointLineThickness * ratio;

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Class.Id);

            var points = GetPoints(box);
            var textLocation = points[0];

            process.Mutate(context =>
            {
                DrawBoundingBox(context, points, color, boxBorderThickness, .1f);

                DrawTextLabel(context, label, textLocation, color, boxBorderThickness, textPadding, textOptions);

                // Draw lines
                for (int i = 0; i < options.Skeleton.Connections.Count; i++)
                {
                    var connection = options.Skeleton.Connections[i];

                    var first = box.Keypoints.ElementAt(connection.First);
                    var second = box.Keypoints.ElementAt(connection.Second);

                    if (first.Confidence < options.KeypointConfidence || second.Confidence < options.KeypointConfidence)
                        continue;

                    var points = new PointF[]
                    {
                        first.Point,
                        second.Point,
                    };

                    var lineColor = options.Skeleton.GetLineColor(i);

                    context.DrawLine(lineColor, lineThickness, points);
                }

                // Draw keypoints
                foreach (var keypoint in box.Keypoints)
                {
                    if (keypoint.Confidence < options.KeypointConfidence)
                        continue;

                    var ellipse = new EllipsePolygon(keypoint.Point, radius);

                    var keypointColor = options.Skeleton.GetKeypointColor(keypoint.Index);

                    context.Fill(keypointColor, ellipse);
                }
            });
        }

        return process;
    }

    public static Image PlotImage(this DetectionResult result, ImageSelector<Rgba32> originImage, DetectionPlottingOptions? options = null)
    {
        options ??= DetectionPlottingOptions.Default;

        var process = originImage.Load(true);

        process.Mutate(x => x.AutoOrient());

        EnsureSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderThickness * ratio;

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Class.Id);

            var points = GetPoints(box);
            var textLocation = points[0]; // The first point is top left

            process.Mutate(context =>
            {
                DrawBoundingBox(context, points, color, thickness, .1f);

                DrawTextLabel(context, label, textLocation, color, thickness, textPadding, textOptions);
            });
        }

        return process;
    }

    public static Image PlotImage(this ObbDetectionResult result, ImageSelector<Rgba32> originImage, DetectionPlottingOptions? options = null)
    {
        options ??= DetectionPlottingOptions.Default;

        var process = originImage.Load(true);

        process.Mutate(x => x.AutoOrient());

        EnsureSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640f;

        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderThickness * ratio;

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Class.Id);

            var points = GetPoints(box);
            var textLocation = points.MinBy(p => p.Y);

            process.Mutate(context =>
            {
                DrawBoundingBox(context, points, color, thickness, .1f);

                DrawTextLabel(context, label, textLocation, color, thickness, textPadding, textOptions);
            });
        }

        return process;
    }

    public static Image PlotImage(this SegmentationResult result, ImageSelector<Rgba32> originImage, SegmentationPlottingOptions? options = null)
    {
        options ??= SegmentationPlottingOptions.Default;

        var process = originImage.Load(true);

        EnsureSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderThickness * ratio;

        #region Draw Masks

        using var masksLayer = new Image<Rgba32>(size.Width, size.Height);
        using var contoursLayer = new Image<Rgba32>(size.Width, size.Height);

        foreach (var box in result.Boxes)
        {
            var color = options.ColorPalette.GetColor(box.Class.Id);

            using var mask = new Image<Rgba32>(box.Bounds.Width, box.Bounds.Height);

            for (int x = 0; x < box.Mask.Width; x++)
            {
                for (int y = 0; y < box.Mask.Height; y++)
                {
                    var value = box.Mask[x, y];

                    if (value > options.MaskConfidence)
                        mask[x, y] = color;
                }
            }

            masksLayer.Mutate(x => x.DrawImage(mask, box.Bounds.Location, 1F));

            if (options.ContoursThickness > 0F)
            {
                using var contours = CreateContours(mask, color, options.ContoursThickness * ratio);
                contoursLayer.Mutate(x => x.DrawImage(contours, box.Bounds.Location, 1F));
            }
        }

        process.Mutate(x => x.DrawImage(masksLayer, .4F));
        process.Mutate(x => x.DrawImage(contoursLayer, 1F));

        #endregion

        #region Draw Boxes

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Class.Id);

            var points = GetPoints(box);
            var textLocation = points[0];

            process.Mutate(context =>
            {
                DrawBoundingBox(context, points, color, thickness, .1f);

                DrawTextLabel(context, label, textLocation, color, thickness, textPadding, textOptions);
            });
        }

        #endregion

        return process;
    }

    public static Image PlotImage(this ClassificationResult result, ImageSelector<Rgba32> originImage, ClassificationPlottingOptions? options = null)
    {
        options ??= ClassificationPlottingOptions.Default;

        var process = originImage.Load(true);

        EnsureSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));

        var label = result.ToString();

        var classId = result.TopClass.Name.Id;

        var fill = options.FillColorPalette.GetColor(classId);
        var border = options.BorderColorPalette.GetColor(classId);

        var pen = new SolidPen(border, options.BorderThickness * ratio);
        var brush = new SolidBrush(fill);
        var location = new PointF(options.XOffset * ratio, options.YOffset * ratio);

        process.Mutate(x => x.DrawText(label, textOptions.Font, brush, pen, location));

        return process;
    }

    #region Private Methods

    private static void DrawBoundingBox(IImageProcessingContext context, PointF[] points, Color color, float thickness, float opacity)
    {
        var polygon = new Polygon(points);

        context.Draw(color, thickness, polygon);

        if (opacity > 0f)
        {
            context.Fill(color.WithAlpha(opacity), polygon);
        }
    }

    private static void DrawTextLabel(IImageProcessingContext context, string text, PointF location, Color color, float thickness, float padding, TextOptions options)
    {
        var rendered = TextMeasurer.MeasureSize(text, options);
        var renderedSize = new Size((int)(rendered.Width + padding), (int)rendered.Height);

        location.Offset(0, -renderedSize.Height);

        var textLocation = new PointF(location.X + padding / 2, location.Y);

        var textBoxPolygon = new RectangularPolygon(location, renderedSize);

        context.Fill(color, textBoxPolygon);
        context.Draw(color, thickness, textBoxPolygon);

        context.DrawText(text, options.Font, Color.White, textLocation);
    }

    private static Image<Rgba32> CreateContours(this Image source, Color color, float thickness)
    {
        var contours = ImageContoursDetector.FindContours(source);

        var result = new Image<Rgba32>(source.Width, source.Height);

        foreach (var points in contours)
        {
            if (points.Count < 2)
            {
                continue;
            }

            var pathBuilder = new PathBuilder();
            pathBuilder.AddLines(points.Select(x => (PointF)x));

            var path = pathBuilder.Build();

            result.Mutate(x =>
            {
                x.Draw(color, thickness, path);
            });
        }

        return result;
    }

    private static PointF[] GetPoints(BoundingBox box)
    {
        var rect = box.Bounds;

        return
        [
            new PointF(rect.Left, rect.Top),
            new PointF(rect.Right, rect.Top),
            new PointF(rect.Right, rect.Bottom),
            new PointF(rect.Left, rect.Bottom),
        ];
    }

    private static PointF[] GetPoints(ObbBoundingBox box)
    {
        var points = box.GetCornerPoints();

        return [.. points.Select(point => new PointF(point.X, point.Y))];
    }

    private static void EnsureSize(Size origin, Size result)
    {
        if (origin != result)
        {
            throw new InvalidOperationException("Original image size must to be equals to prediction result image size");
        }
    }

    #endregion
}