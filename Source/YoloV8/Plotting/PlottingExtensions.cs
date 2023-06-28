using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;

using Compunet.YoloV8.Data;

namespace Compunet.YoloV8.Plotting;

public static class PlottingExtensions
{
    #region Privates

    private const string _fontName = "Arial";
    private const float _fontSize = 12F;
    private const float _textPadding = 5F;
    private const float _boxThickness = 1F;

    private const float _keypointRadius = 3F;
    private const float _keypointLineWidth = 1.5F;

    private const float _keypointConfidence = .5F;

    private static readonly ISkeleton _humanSkeleton = new HumanSkeleton();

    #endregion

    #region Pose

    public static Image PlotImage(this IPoseResult result, Image origin) => PlotImage(result, origin, _humanSkeleton);

    public static Image PlotImage(this IPoseResult result, Image origin, ISkeleton skeleton)
    {
        var process = origin.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(_fontName, _fontSize * ratio));

        var textPadding = _textPadding * ratio;

        var thickness = _boxThickness * ratio;

        var radius = _keypointRadius * ratio;
        var lineWidth = _keypointLineWidth * ratio;

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = GetColor(box.Class.Id);

            process.Mutate(context =>
            {
                DrawBoundingBox(context, box.Rectangle, color, thickness, label, textOptions, textPadding);

                // drawing lines
                for (int i = 0; i < skeleton.Connections.Count; i++)
                {
                    var connection = skeleton.Connections[i];

                    IKeypoint first = box.Keypoints[connection.First];
                    IKeypoint second = box.Keypoints[connection.Second];

                    if (first.Confidence < _keypointConfidence || second.Confidence < _keypointConfidence)
                        continue;

                    var points = new PointF[]
                    {
                        first.Point,
                        second.Point,
                    };

                    var lineColor = skeleton.GetLineColor(i);

                    context.DrawLines(lineColor, lineWidth, points);
                }

                // drawing keypoints
                for (int i = 0; i < box.Keypoints.Count; i++)
                {
                    var keypoint = box.Keypoints[i];

                    if (keypoint.Confidence < _keypointConfidence)
                        continue;

                    var ellipse = new EllipsePolygon(keypoint.Point, radius);

                    var keypointColor = skeleton.GetKeypointColor(keypoint.Index);

                    context.Fill(keypointColor, ellipse);
                }
            });
        }

        return process;
    }

    #endregion

    #region Detection

    public static Image PlotImage(this IDetectionResult result, Image origin)
    {
        var process = origin.CloneAs<Rgba32>();
        process.Mutate(x => x.AutoOrient());

        CheckSize(process.Size, result.Image);

        var size = result.Image;

        var ratio = Math.Max(size.Width, size.Height) / 640F;

        var textOptions = new TextOptions(SystemFonts.CreateFont(_fontName, _fontSize * ratio));

        var textPadding = _textPadding * ratio;

        var thickness = _boxThickness * ratio;

        foreach (var box in result.Boxes)
        {
            var label = $"{box.Class.Name} {box.Confidence:N}";
            var color = GetColor(box.Class.Id);

            process.Mutate(context =>
            {
                DrawBoundingBox(context, box.Rectangle, color, thickness, label, textOptions, textPadding);
            });
        }

        return process;
    }

    #endregion

    #region Private Methods

    private static void DrawBoundingBox(IImageProcessingContext context,
                                        Rectangle rectangle,
                                        Color color,
                                        float thickness,
                                        string label,
                                        TextOptions textOptions,
                                        float textPadding)
    {
        var polygon = new RectangularPolygon(rectangle);

        context.Draw(color, thickness, polygon);
        context.Fill(color.WithAlpha(.1F), polygon);

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

    private static Color GetColor(int index)
    {
        string hex = (index % 20) switch
        {
            0 => "FF3838",
            1 => "FF9D97",
            2 => "FF701F",
            3 => "FFB21D",
            4 => "CFD231",
            5 => "48F90A",
            6 => "92CC17",
            7 => "3DDB86",
            8 => "1A9334",
            9 => "00D4BB",
            10 => "2C99A8",
            11 => "00C2FF",
            12 => "344593",
            13 => "6473FF",
            14 => "0018EC",
            15 => "8438FF",
            16 => "520085",
            17 => "CB38FF",
            18 => "FF95C8",
            19 => "FF37C7",
            _ => throw new IndexOutOfRangeException()
        };

        return Color.ParseHex(hex);
    }

    #endregion
}