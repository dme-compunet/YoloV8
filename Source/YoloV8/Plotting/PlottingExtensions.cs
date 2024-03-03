namespace Compunet.YoloV8.Plotting;

public static class PlottingExtensions
{
    #region Pose

    public static Image PlotImage(this PoseResult result, ImageSelector<Rgba32> originImage) => PlotImage(result, originImage, PosePlottingOptions.Default);

    public static Image PlotImage(this PoseResult result, ImageSelector<Rgba32> originImage, PosePlottingOptions options)
    {
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

            process.Mutate(context =>
            {
                DrawBoundingBox(context, box.Bounds, color, boxBorderThickness, .1F, label, textOptions, textPadding);

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

    #endregion

    #region Detection

    public static Image PlotImage(this DetectionResult result, ImageSelector<Rgba32> originImage) => result.PlotImage(originImage, DetectionPlottingOptions.Default);

    public static Image PlotImage(this DetectionResult result, ImageSelector<Rgba32> originImage, DetectionPlottingOptions options)
    {
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

            process.Mutate(context =>
            {
                DrawBoundingBox(context, box.Bounds, color, thickness, .1F, label, textOptions, textPadding);
            });
        }

        return process;
    }

    #endregion

    #region Segmentation

    public static Image PlotImage(this SegmentationResult result, ImageSelector<Rgba32> originImage) => result.PlotImage(originImage, SegmentationPlottingOptions.Default);

    public static Image PlotImage(this SegmentationResult result, ImageSelector<Rgba32> originImage, SegmentationPlottingOptions options)
    {
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

            process.Mutate(context =>
            {
                DrawBoundingBox(context,
                                box.Bounds,
                                color,
                                thickness,
                                0F,
                                label,
                                textOptions,
                                textPadding);
            });
        }

        #endregion

        return process;
    }

    #endregion

    #region Classification

    public static Image PlotImage(this ClassificationResult result, ImageSelector<Rgba32> originImage) => PlotImage(result, originImage, ClassificationPlottingOptions.Default);

    public static Image PlotImage(this ClassificationResult result, ImageSelector<Rgba32> originImage, ClassificationPlottingOptions options)
    {
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

    #endregion

    #region Private Methods

    private static void DrawBoundingBox(IImageProcessingContext context,
                                        Rectangle bounds,
                                        Color color,
                                        float borderThickness,
                                        float fillOpacity,
                                        string labelText,
                                        TextOptions textOptions,
                                        float textPadding)
    {
        var polygon = new RectangularPolygon(bounds);

        context.Draw(color, borderThickness, polygon);

        if (fillOpacity > 0F)
            context.Fill(color.WithAlpha(fillOpacity), polygon);

        var rendered = TextMeasurer.MeasureSize(labelText, textOptions);
        var renderedSize = new Size((int)(rendered.Width + textPadding), (int)rendered.Height);

        var location = bounds.Location;

        location.Offset(0, -renderedSize.Height);

        //var textLocation = new Point((int)(location.X + textPadding / 2), location.Y);
        var textLocation = new PointF(location.X + textPadding / 2, location.Y);

        var textBoxPolygon = new RectangularPolygon(location, renderedSize);

        context.Fill(color, textBoxPolygon);
        context.Draw(color, borderThickness, textBoxPolygon);

        context.DrawText(labelText, textOptions.Font, Color.White, textLocation);
    }

    private static Image CreateContours(this Image source, Color color, float thickness)
    {
        var contours = ImageContoursDetector.FindContours(source);

        var result = new Image<Rgba32>(source.Width, source.Height);

        foreach (var points in contours)
        {
            if (points.Count < 2)
                continue;

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


    private static void EnsureSize(Size origin, Size result)
    {
        if (origin != result)
            throw new InvalidOperationException("Original image size must to be equals to prediction result image size");
    }

    #endregion
}