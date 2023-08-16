using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;

using Compunet.YoloV8.Data;
using Compunet.YoloV8.Extensions;
using System.Data;

namespace Compunet.YoloV8.Plotting;

public static class PlottingExtensions
{
    #region Pose

    public static Image PlotImage(this IPoseResult result, Image originImage) => PlotImage(result, originImage, PosePlottingOptions.Default);

    public static Image PlotImage(this IPoseResult result, Image originImage, PosePlottingOptions options)
    {
        var process = originImage.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(options.FontName, options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderThickness * ratio;

        var radius = options.KeypointRadius * ratio;
        var lineThickness = options.KeypointLineThickness * ratio;

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Class.Id);

            process.Mutate(context =>
            {
                DrawBoundingBox(context, box.Rectangle, color, thickness, .1F, label, textOptions, textPadding);

                // drawing lines
                for (int i = 0; i < options.Skeleton.Connections.Count; i++)
                {
                    var connection = options.Skeleton.Connections[i];

                    IKeypoint first = box.Keypoints[connection.First];
                    IKeypoint second = box.Keypoints[connection.Second];

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

                // drawing keypoints
                for (int i = 0; i < box.Keypoints.Count; i++)
                {
                    var keypoint = box.Keypoints[i];

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

    public static Image PlotImage(this IDetectionResult result, Image originImage) => result.PlotImage(originImage, PlottingOptions.Default);

    public static Image PlotImage(this IDetectionResult result, Image originImage, PlottingOptions options)
    {
        var process = originImage.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(options.FontName, options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderThickness * ratio;

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Class.Id);

            process.Mutate(context =>
            {
                DrawBoundingBox(context, box.Rectangle, color, thickness, .1F, label, textOptions, textPadding);
            });
        }

        return process;
    }

    #endregion

    #region Segmentation

    public static Image PlotImage(this ISegmentationResult result, Image originImage) => result.PlotImage(originImage, SegmentationPlottingOptions.Default);

    public static Image PlotImage(this ISegmentationResult result, Image originImage, SegmentationPlottingOptions options)
    {
        var process = originImage.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(options.FontName, options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderThickness * ratio;

        #region Draw Masks

        using var masks = new Image<Rgba32>(size.Width, size.Height);
        using var contours = new Image<Rgba32>(size.Width, size.Height);

        for (int i = 0; i < result.Boxes.Count; i++)
        {
            var box = result.Boxes[i];
            var color = options.ColorPalette.GetColor(box.Class.Id);

            using var mask = new Image<Rgba32>(box.Rectangle.Width, box.Rectangle.Height);

            for (int x = 0; x < box.Mask.Width; x++)
            {
                for (int y = 0; y < box.Mask.Height; y++)
                {
                    var value = box.Mask[x, y];

                    if (value > options.MaskConfidence)
                        mask[x, y] = color;
                }
            }

            masks.Mutate(x => x.DrawImage(mask, box.Rectangle.Location, 1F));

            if (options.ContoursThickness > 0F)
            {
                using var contour = CreateContours(mask, color, options.ContoursThickness * ratio);
                contours.Mutate(x => x.DrawImage(contour, box.Rectangle.Location, 1F));
            }
        }

        process.Mutate(x => x.DrawImage(masks, .4F));
        process.Mutate(x => x.DrawImage(contours, 1F));

        #endregion

        #region Draw Boxes

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = options.ColorPalette.GetColor(box.Class.Id);

            process.Mutate(context =>
            {
                DrawBoundingBox(context,
                                box.Rectangle,
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

    public static Image PlotImage(this IClassificationResult result, Image originImage, ClassificationPlottingOptions options)
    {
        var process = originImage.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(options.FontName, options.FontSize * ratio));

        var label = $"{result.Class.Name} {result.Confidence:N}";
        var fill = options.FillColorPalette.GetColor(result.Class.Id);
        var border = options.BorderColorPalette.GetColor(result.Class.Id);

        var pen = new Pen(border, options.BorderThickness * ratio);
        var brush = new SolidBrush(fill);
        var location = new PointF(options.XOffset * ratio, options.YOffset * ratio);

        process.Mutate(x => x.DrawText(label, textOptions.Font, brush, pen, location));

        return process;
    }

    #endregion

    #region Private Methods

    private static void DrawBoundingBox(IImageProcessingContext context,
                                        Rectangle rectangle,
                                        Color color,
                                        float borderThickness,
                                        float fillOpacity,
                                        string labelText,
                                        TextOptions textOptions,
                                        float textPadding)
    {
        var polygon = new RectangularPolygon(rectangle);

        context.Draw(color, borderThickness, polygon);

        if (fillOpacity > 0F)
            context.Fill(color.WithAlpha(fillOpacity), polygon);

        var rendered = TextMeasurer.Measure(labelText, textOptions);
        var renderedSize = new Size((int)(rendered.Width + textPadding), (int)rendered.Height);

        var location = rectangle.Location;

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
        var contours = source.GetContours();

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


    private static void CheckSize(Size origin, Size result)
    {
        if (origin != result)
            throw new InvalidOperationException("The size of the original image is different from the size of the image in the result");
    }

    #endregion
}