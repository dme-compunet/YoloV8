namespace Compunet.YoloV8.Plotting;

public class DetectionPlottingOptions : PlottingOptions
{
    public static DetectionPlottingOptions Default { get; } = new DetectionPlottingOptions();

    public float LabelTextXPadding { get; set; }

    public float LabelTextYPadding { get; set; }

    public float BoxBorderThickness { get; set; }

    public ColorPalette ColorPalette { get; set; }

    public DetectionPlottingOptions()
    {
        LabelTextXPadding = 6f;
        LabelTextYPadding = 4f;
        BoxBorderThickness = 1f;
        ColorPalette = ColorPalette.Default;
    }
}