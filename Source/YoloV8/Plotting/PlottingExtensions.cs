using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;

using Compunet.YoloV8.Data;

namespace Compunet.YoloV8.Plotting;

public static class PlottingExtensions
{
    #region Pose

    public static Image PlotImage(this IPoseResult result, Image origin) => PlotImage(result, origin, PosePlottingOptions.Default);

    public static Image PlotImage(this IPoseResult result, Image origin, PosePlottingOptions options)
    {
        var process = origin.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(options.FontName, options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderWith * ratio;

        var radius = options.KeypointRadius * ratio;
        var lineWidth = options.KeypointLineWidth * ratio;

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

                    context.DrawLines(lineColor, lineWidth, points);
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

    public static Image PlotImage(this IDetectionResult result, Image origin) => result.PlotImage(origin, PlottingOptions.Default);

    public static Image PlotImage(this IDetectionResult result, Image origin, PlottingOptions options)
    {
        var process = origin.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(options.FontName, options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderWith * ratio;

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

    public static Image PlotImage(this ISegmentationResult result, Image origin) => result.PlotImage(origin, SegmentationPlottingOptions.Default);

    public static Image PlotImage(this ISegmentationResult result, Image origin, SegmentationPlottingOptions options)
    {
        var process = origin.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(options.FontName, options.FontSize * ratio));

        var textPadding = options.TextHorizontalPadding * ratio;

        var thickness = options.BoxBorderWith * ratio;

        #region Draw Masks

        using var segmentation = new Image<Rgba32>(size.Width, size.Height);

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

            segmentation.Mutate(x => x.DrawImage(mask, box.Rectangle.Location, 1F));
        }

        process.Mutate(x => x.DrawImage(segmentation, .5F));

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

    #region Private Methods

    private static void DrawBoundingBox(IImageProcessingContext context,
                                        Rectangle rectangle,
                                        Color color,
                                        float thickness,
                                        float fillOpacity,
                                        string label,
                                        TextOptions textOptions,
                                        float textPadding)
    {
        var polygon = new RectangularPolygon(rectangle);

        context.Draw(color, thickness, polygon);

        if (fillOpacity > 0F)
            context.Fill(color.WithAlpha(fillOpacity), polygon);

        var rendered = TextMeasurer.Measure(label, textOptions);
        var renderedSize = new Size((int)(rendered.Width + textPadding), (int)rendered.Height);

        var location = rectangle.Location;

        location.Offset(0, -renderedSize.Height);

        var textLocation = new Point((int)(location.X + textPadding / 2), location.Y);

        var textBoxPolygon = new RectangularPolygon(location, renderedSize);

        context.Fill(color, textBoxPolygon);
        context.Draw(color, thickness, textBoxPolygon);

        context.DrawText(label, textOptions.Font, Color.White, textLocation);
    }

    private static void CheckSize(Size origin, Size result)
    {
        if (origin != result)
            throw new InvalidOperationException("The size of the original image is different from the size of the image in the result");
    }

    #endregion
}