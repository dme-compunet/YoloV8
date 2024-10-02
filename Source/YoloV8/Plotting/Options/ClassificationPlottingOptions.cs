namespace Compunet.YoloV8.Plotting;

public class ClassificationPlottingOptions : PlottingOptions
{
    public static ClassificationPlottingOptions Default { get; } = new ClassificationPlottingOptions();

    public Point Location { get; set; }

    public ClassificationPlottingOptions()
    {
        FontSize = 16;
        Location = new Point(30, 30);
    }
}