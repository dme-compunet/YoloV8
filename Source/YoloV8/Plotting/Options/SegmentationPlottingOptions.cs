namespace Compunet.YoloV8.Plotting;

public class SegmentationPlottingOptions : DetectionPlottingOptions
{
    public static new SegmentationPlottingOptions Default { get; } = new SegmentationPlottingOptions();

    public float MaskConfidence { get; set; }

    public float ContoursThickness { get; set; }

    public SegmentationPlottingOptions()
    {
        MaskConfidence = .5F;
        ContoursThickness = 1f;
    }
}