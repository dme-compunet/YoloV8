namespace Compunet.YoloV8.Plotting;

public class PlottingOptions
{
    internal static PlottingOptions Default { get; } = new PlottingOptions();

    public string FontName { get; set; }

    public float FontSize { get; set; }

    public float TextHorizontalPadding { get; set; }

    public float BoxBorderWith { get; set; }

    public ColorPalette ColorPalette { get; set; }

    public PlottingOptions()
    {
        FontName = "Arial";
        FontSize = 12F;
        TextHorizontalPadding = 5F;
        BoxBorderWith = 1F;
        ColorPalette = ColorPalette.Default;
    }
}