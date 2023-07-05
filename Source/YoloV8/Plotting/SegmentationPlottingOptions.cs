namespace Compunet.YoloV8.Plotting;

public class SegmentationPlottingOptions : PlottingOptions
{
    internal static new SegmentationPlottingOptions Default { get; } = new SegmentationPlottingOptions();

    public float MaskConfidence { get; set; }

    public SegmentationPlottingOptions()
    {
        MaskConfidence = .5F;
    }
}