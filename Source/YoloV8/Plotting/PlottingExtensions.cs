using System.Numerics;

namespace Compunet.YoloV8.Plotting;

public static class PlottingExtensions
{
    public static Image PlotImage(this YoloResult<Pose> result, Image image, PosePlottingOptions? options = null)
    {
        options ??= PosePlottingOptions.Default;

        var target = image.CloneAs<Rgba32>();

        target.AutoOrient();

        ValidateSize(image.Size, result.ImageSize);

        var ratio = GetRatio(image.Size);
        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));
        var textPadding = new Vector2(options.LabelTextXPadding, options.LabelTextYPadding) * ratio;
        var boxBorderThickness = options.BoxBorderThickness * ratio;
        var radius = options.KeypointRadius * ratio;
        var lineThickness = options.KeypointLineThickness * ratio;

        foreach (var box in result)
        {
            var label = $"{box.Name.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Name.Id);

            var points = GetPoints(box);
            var textLocation = points[0];

            target.Mutate(context =>
            {
                context.DrawBox(points, color, boxBorderThickness, .1f);
                context.DrawLabel(label, textLocation, color, boxBorderThickness, textPadding, textOptions);

                // Draw lines
                for (var i = 0; i < options.Skeleton.Connections.Length; i++)
                {
                    var connection = options.Skeleton.Connections[i];

                    var first = box[connection.First];
                    var second = box[connection.Second];

                    if (first.Confidence < options.KeypointConfidence || second.Confidence < options.KeypointConfidence)
                    {
                        continue;
                    }

                    var points = new PointF[]
                    {
                        first.Point,
                        second.Point,
                    };

                    var lineColor = options.Skeleton.GetLineColor(i);

                    context.DrawLine(lineColor, lineThickness, points);
                }

                // Draw keypoints
                foreach (var keypoint in box)
                {
                    if (keypoint.Confidence < options.KeypointConfidence)
                    {
                        continue;
                    }

                    var ellipse = new EllipsePolygon(keypoint.Point, radius);

                    var keypointColor = options.Skeleton.GetKeypointColor(keypoint.Index);

                    context.Fill(keypointColor, ellipse);
                }
            });
        }

        return target;
    }

    public static Image PlotImage(this YoloResult<Detection> result, Image image, DetectionPlottingOptions? options = null)
    {
        options ??= DetectionPlottingOptions.Default;

        var target = image.CloneAs<Rgba32>();

        target.AutoOrient();

        ValidateSize(image.Size, result.ImageSize);

        var ratio = GetRatio(image.Size);

        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio))
        {
            VerticalAlignment = VerticalAlignment.Center,
        };

        var thickness = options.BoxBorderThickness * ratio;
        var textPadding = new Vector2(options.LabelTextXPadding, options.LabelTextYPadding) * ratio;

        foreach (var box in result)
        {
            var label = $"{box.Name.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Name.Id);

            var points = GetPoints(box);
            var textLocation = points[0]; // The first point is top left

            target.Mutate(context =>
            {
                context.DrawBox(points, color, thickness, .1f);
                context.DrawLabel(label, textLocation, color, thickness, textPadding, textOptions);
            });
        }

        return target;
    }

    public static Image PlotImage(this YoloResult<ObbDetection> result, Image image, DetectionPlottingOptions? options = null)
    {
        options ??= DetectionPlottingOptions.Default;

        var target = image.CloneAs<Rgba32>();

        target.AutoOrient();

        ValidateSize(target.Size, result.ImageSize);

        var ratio = GetRatio(image.Size);
        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));
        var textPadding = new Vector2(options.LabelTextXPadding, options.LabelTextYPadding) * ratio;
        var thickness = options.BoxBorderThickness * ratio;

        foreach (var box in result)
        {
            var label = $"{box.Name.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Name.Id);

            var points = GetPoints(box);
            var textLocation = points.MinBy(p => p.Y);

            target.Mutate(context =>
            {
                context.DrawBox(points, color, thickness, .1f);
                context.DrawLabel(label, textLocation, color, thickness, textPadding, textOptions);
            });
        }

        return target;
    }

    public static Image PlotImage(this YoloResult<Segmentation> result, Image image, SegmentationPlottingOptions? options = null)
    {
        options ??= SegmentationPlottingOptions.Default;

        var target = image.CloneAs<Rgba32>();

        target.AutoOrient();

        ValidateSize(target.Size, result.ImageSize);

        var size = result.ImageSize;
        var ratio = GetRatio(image.Size);
        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));
        var textPadding = new Vector2(options.LabelTextXPadding, options.LabelTextYPadding) * ratio;
        var thickness = options.BoxBorderThickness * ratio;

        #region Draw Masks

        using var masksLayer = new Image<Rgba32>(size.Width, size.Height);
        using var contoursLayer = new Image<Rgba32>(size.Width, size.Height);

        foreach (var box in result)
        {
            var color = options.ColorPalette.GetColor(box.Name.Id);

            using var mask = new Image<Rgba32>(box.Bounds.Width, box.Bounds.Height);

            for (var x = 0; x < box.Mask.Width; x++)
            {
                for (var y = 0; y < box.Mask.Height; y++)
                {
                    var value = box.Mask[x, y];

                    if (value > options.MaskConfidence)
                    {
                        mask[x, y] = color;
                    }
                }
            }

            masksLayer.Mutate(x => x.DrawImage(mask, box.Bounds.Location, 1F));

            if (options.ContoursThickness > 0f)
            {
                using var contours = CreateContours(mask, color, options.ContoursThickness * ratio);
                contoursLayer.Mutate(x => x.DrawImage(contours, box.Bounds.Location, 1f));
            }
        }

        target.Mutate(x => x.DrawImage(masksLayer, .4F));
        target.Mutate(x => x.DrawImage(contoursLayer, 1F));

        #endregion

        #region Draw Boxes

        foreach (var box in result)
        {
            var label = $"{box.Name.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Name.Id);

            var points = GetPoints(box);
            var textLocation = points[0];

            target.Mutate(context =>
            {
                context.DrawBox(points, color, thickness, .1f);
                context.DrawLabel(label, textLocation, color, thickness, textPadding, textOptions);
            });
        }

        #endregion

        return target;
    }

    public static Image PlotImage(this YoloResult<Classification> result, Image image, ClassificationPlottingOptions? options = null)
    {
        options ??= ClassificationPlottingOptions.Default;

        var target = image.CloneAs<Rgba32>();

        target.AutoOrient();

        ValidateSize(target.Size, result.ImageSize);

        var ratio = GetRatio(image.Size);
        var textOptions = new TextOptions(options.FontFamily.CreateFont(options.FontSize * ratio));

        var label = result.ToString();
        var classId = result.GetTopClass().Name.Id;

        var fill = options.FillColorPalette.GetColor(classId);
        var border = options.BorderColorPalette.GetColor(classId);

        var pen = new SolidPen(border, options.BorderThickness * ratio);
        var brush = new SolidBrush(fill);
        var location = new PointF(options.XOffset * ratio, options.YOffset * ratio);

        target.Mutate(x => x.DrawText(label, textOptions.Font, brush, pen, location));

        return target;
    }

    #region Private Methods

    private static void DrawBox(this IImageProcessingContext context, PointF[] points, Color color, float thickness, float opacity)
    {
        var polygon = new Polygon(points);

        context.Draw(color, thickness, polygon);

        if (opacity > 0f)
        {
            context.Fill(color.WithAlpha(opacity), polygon);
        }
    }

    private static void DrawLabel(this IImageProcessingContext context, string text, PointF location, Color color, float thickness, Vector2 padding, TextOptions options)
    {
        var xPadding = padding.X;
        var yPadding = padding.Y;

        var rendered = TextMeasurer.MeasureBounds(text, options);
        var labelSize = new SizeF(rendered.Width + xPadding, options.Font.Size + yPadding);

        location.Offset(0, -labelSize.Height);

        var textLocation = new PointF(location.X + xPadding / 2, location.Y + yPadding / 2);
        var textBoxPolygon = new RectangularPolygon(location, labelSize);

        // Fix text position
        textLocation.Offset(0, -(yPadding * .1f));

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
            if (points.Length < 2)
            {
                continue;
            }

            var path = new PathBuilder().AddLines(points.Select(point => (PointF)point)).Build();

            result.Mutate(x =>
            {
                x.Draw(color, thickness, path);
            });
        }

        return result;
    }

    private static PointF[] GetPoints(Detection box)
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

    private static PointF[] GetPoints(ObbDetection box)
    {
        var points = box.GetCornerPoints();

        return [.. points.Select(point => new PointF(point.X, point.Y))];
    }

    private static void ValidateSize(Size origin, Size result)
    {
        if (origin != result)
        {
            throw new InvalidOperationException("Original image size must to be equals to prediction result image size");
        }
    }

    private static float GetRatio(Size size)
    {
        return Math.Max(size.Width, size.Height) / 640f;
    }

    #endregion
}