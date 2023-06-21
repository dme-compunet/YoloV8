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

    private static readonly string[] _poseColors = new[]
    {
        "FF8000",
        "FF9933",
        "FFB266",
        "E6E600",
        "FF99FF",
        "99CCFF",
        "FF66FF",
        "FF33FF",
        "66B2FF",
        "3399FF",
        "FF9999",
        "FF6666",
        "FF3333",
        "99FF99",
        "66FF66",
        "33FF33",
        "00FF00",
        "0000FF",
        "FF0000",
        "FFFFFF",
    };

    private const string _fontName = "Arial";
    private const float _fontSize = 12F;
    private const float _textPadding = 5F;
    private const float _boxThickness = 1F;

    private const float _keypointRadius = 3F;
    private const float _keypointLineWidth = 1.5F;

    private static readonly int[,] _skeleton = new[,]
    {
        {16, 14},
        {14, 12},
        {17, 15},
        {15, 13},
        {12, 13},
        {6, 12},
        {7, 13},
        {6, 7},
        {6, 8},
        {7, 9},
        {8, 10},
        {9, 11},
        {2, 3},
        {1, 2},
        {1, 3},
        {2, 4},
        {3, 5},
        {4, 6},
        {5, 7}
    };

    private static readonly int[] _keypointColorMap = new[]
    {
        16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9
    };

    private static readonly int[] _keypointLineColorMap = new[]
    {
        9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16
    };

    #endregion

    #region CreatePlottingLayer

    public static Image CreatePlottingLayer(IPoseResult result)
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

        foreach (var person in result.Persons)
        {
            var text = $"person {person.Confidence:N}";

            canvas.Mutate(x =>
            {
                DrawBoundingBox(x,
                                person.Rectangle,
                                thickness,
                                color);

                DrawTextLabel(x,
                              text,
                              textOptions,
                              person.Rectangle.Location,
                              padding,
                              color,
                              thickness);

                // drawing lines
                for (int i = 0; i < _skeleton.GetLength(0); i++)
                {
                    var first = _skeleton[i, 0];
                    var second = _skeleton[i, 1];

                    IKeypoint? firstKp = person.GetKeypoint(first);
                    IKeypoint? secondKp = person.GetKeypoint(second);

                    if (firstKp is null || secondKp is null)
                        continue;

                    var points = new PointF[]
                    {
                        firstKp.Point,
                        secondKp.Point,
                    };

                    var lineColor = GetPoseColor(i, true);

                    x.DrawLines(lineColor, lineWidth, points);
                }

                // drawing keypoints
                for (int i = 0; i < person.Keypoints.Count; i++)
                {
                    var keypoint = person.Keypoints[i];

                    var ellipse = new EllipsePolygon(keypoint.Point, radius);

                    var kpColor = GetPoseColor(keypoint.Id - 1, false);

                    x.Fill(kpColor, ellipse);
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

    public static Image PlotImage(Image origin, IPoseResult result)
    {
        var processed = origin.Clone(x => x.AutoOrient());

        if (processed.Size != result.Image)
            throw new InvalidOperationException("The size of the original image is different from the size of the image in the result");

        var plottinLayer = CreatePlottingLayer(result);

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

    private static Color GetPoseColor(int index, bool line)
    {
        index = line ? _keypointLineColorMap[index]
                     : _keypointColorMap[index];

        var hex = _poseColors[index];

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