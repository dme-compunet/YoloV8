using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;

using Compunet.YoloV8.Data;

namespace Compunet.YoloV8.Plotting;

internal static class PlottingUtilities
{
    #region Privates

    private static readonly string[] _colors = new[]
    {
        "FF3838",
        "FF9D97",
        "FF701F",
        "FFB21D",
        "CFD231",
        "48F90A",
        "92CC17",
        "3DDB86",
        "1A9334",
        "00D4BB",
        "2C99A8",
        "00C2FF",
        "344593",
        "6473FF",
        "0018EC",
        "8438FF",
        "520085",
        "CB38FF",
        "FF95C8",
        "FF37C7",
    };

    private const string _fontName = "Arial";
    private const float _fontSize = 12F;
    private const float _textPadding = 5F;
    private const float _boxThickness = 1F;

    private const float _keypointRadius = 3F;
    private const float _keypointLineWidth = 1.5F;

    private static readonly ISkeleton _humanSkeleton = new HumanSkeleton();

    #endregion

    #region CreatePlottingLayer

    public static Image CreatePlottingLayer(IPoseResult result, ISkeleton skeleton)
    {
        var size = result.Image;

        var canvas = new Image<Rgba32>(size.Width, size.Height);

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(_fontName, _fontSize * ratio));

        var color = GetColor(0);
        var thickness = _boxThickness * ratio;
        var padding = _textPadding * ratio;

        var radius = _keypointRadius * ratio;
        var lineWidth = _keypointLineWidth * ratio;

        foreach (var box in result.Boxes)
        {
            var text = $"{box.Class.Name} {box.Confidence:N}";

            canvas.Mutate(x =>
            {
                DrawBoundingBox(x,
                                box.Rectangle,
                                thickness,
                                color);

                DrawTextLabel(x,
                              text,
                              textOptions,
                              box.Rectangle.Location,
                              padding,
                              color,
                              thickness);

                // drawing lines
                for (int i = 0; i < skeleton.Connections.Count; i++)
                {
                    var connection = skeleton.Connections[i];

                    IKeypoint? firstKp = box.GetKeypoint(connection.First);
                    IKeypoint? secondKp = box.GetKeypoint(connection.Second);

                    if (firstKp is null || secondKp is null)
                        continue;

                    var points = new PointF[]
                    {
                        firstKp.Point,
                        secondKp.Point,
                    };

                    var lineColor = skeleton.GetLineColor(i);

                    x.DrawLines(lineColor, lineWidth, points);
                }

                // drawing keypoints
                for (int i = 0; i < box.Keypoints.Count; i++)
                {
                    var keypoint = box.Keypoints[i];

                    var ellipse = new EllipsePolygon(keypoint.Point, radius);

                    var keypointColor = skeleton.GetKeypointColor(keypoint.Index);

                    x.Fill(keypointColor, ellipse);
                }
            });
        }

        return canvas;
    }

    public static Image CreatePlottingLayer(IDetectionResult result)
    {
        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var canvas = new Image<Rgba32>(size.Width, size.Height);

        var textOptions = new TextOptions(SystemFonts.CreateFont(_fontName, _fontSize * ratio));

        var thickness = _boxThickness * ratio;
        var padding = _textPadding * ratio;

        foreach (var box in result.Boxes)
        {
            var text = $"{box.Class.Name} {box.Confidence:N}";
            var color = GetColor(box.Class.Id);

            canvas.Mutate(x =>
            {
                DrawBoundingBox(x, box.Rectangle, thickness, color);

                DrawTextLabel(x, text, textOptions, box.Rectangle.Location, padding, color, thickness);
            });
        }

        return canvas;
    }

    #endregion

    #region PlotImage

    public static Image PlotImage(Image origin, IPoseResult result) => PlotImage(origin, result, _humanSkeleton);

    public static Image PlotImage(Image origin, IPoseResult result, ISkeleton skeleton)
    {
        var processed = origin.Clone(x => x.AutoOrient());

        if (processed.Size != result.Image)
            throw new InvalidOperationException("The size of the original image is different from the size of the image in the result");

        var plottinLayer = CreatePlottingLayer(result, skeleton);

        processed.Mutate(x =>
        {
            x.DrawImage(plottinLayer, 1F);
        });

        return processed;
    }

    public static Image PlotImage(Image origin, IDetectionResult result)
    {
        var processed = origin.Clone(x => x.AutoOrient());

        if (processed.Size != result.Image)
            throw new InvalidOperationException();

        var plottinLayer = CreatePlottingLayer(result);

        processed.Mutate(x =>
        {
            x.DrawImage(plottinLayer, 1F);
        });

        return processed;
    }

    #endregion

    #region Private Methods

    private static Color GetColor(int index)
    {
        index %= _colors.Length;
        var hex = _colors[index];

        return Color.ParseHex(hex);
    }

    private static void DrawBoundingBox(IImageProcessingContext context,
                                        Rectangle rectangle,
                                        float thickness,
                                        Color color)
    {
        var polygon = new RectangularPolygon(rectangle);

        context.Draw(color, thickness, polygon);
        context.Fill(color.WithAlpha(.1F), polygon);
    }

    private static void DrawTextLabel(IImageProcessingContext context,
                                      string text,
                                      TextOptions options,
                                      PointF location,
                                      float xPadding,
                                      Color background,
                                      float thickness)
    {
        var rendered = TextMeasurer.Measure(text, options);
        var renderedSize = new SizeF(rendered.Width + xPadding, rendered.Height);

        location.Offset(0F, -renderedSize.Height);

        var textLocation = new PointF(location.X + xPadding / 2, location.Y);

        var textBoxPolygon = new RectangularPolygon(location, renderedSize);

        context.Fill(background, textBoxPolygon);
        context.Draw(background, thickness, textBoxPolygon);

        context.DrawText(text, options.Font, Color.White, textLocation);
    }

    #endregion
}