namespace Compunet.YoloV8.Plotting;

public class ClassificationPlottingOptions : PlottingOptions
{
    public static ClassificationPlottingOptions Default { get; } = new ClassificationPlottingOptions();

    public ColorPalette FillColorPalette { get; set; }

    public ColorPalette BorderColorPalette { get; set; }

    public float BorderThickness { get; set; }

    public float XOffset { get; set; }

    public float YOffset { get; set; }

    public ClassificationPlottingOptions()
    {
        FontSize = 40F;
        FillColorPalette = new ColorPalette("CCCCCC");
        BorderColorPalette = new ColorPalette("333333");
        BorderThickness = 1F;
        XOffset = 30F;
        YOffset = 30F;
    }
}