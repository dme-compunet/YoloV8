namespace Compunet.YoloV8.Plotting;

public abstract class PlottingOptions
{
    public FontFamily FontFamily { get; set; }

    public float FontSize { get; set; }

    public PlottingOptions()
    {
        FontFamily = GetDefaultFontFamily();
        FontSize = 12F;
    }

    private static FontFamily GetDefaultFontFamily()
    {
        if (OperatingSystem.IsWindows())
            return SystemFonts.Get("Arial");

        if (OperatingSystem.IsAndroid())
            return SystemFonts.Get("Robot");

        return SystemFonts.Families.First();
    }
}