namespace Compunet.YoloV8.Plotting;

public class PlottingOptions
{
    public static PlottingOptions Default { get; } = new PlottingOptions();

    public string FontName { get; set; }

    public float FontSize { get; set; }

    public float TextHorizontalPadding { get; set; }

    public float BoxBorderThickness { get; set; }

    public ColorPalette ColorPalette { get; set; }

    public PlottingOptions()
    {
        FontName = "Arial";
        FontSize = 12F;
        TextHorizontalPadding = 5F;
        BoxBorderThickness = 1F;
        ColorPalette = ColorPalette.Default;
    }
}