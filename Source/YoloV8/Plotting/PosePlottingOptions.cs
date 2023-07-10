namespace Compunet.YoloV8.Plotting;

public class PosePlottingOptions : PlottingOptions
{
    internal static new PosePlottingOptions Default { get; } = new PosePlottingOptions();

    public ISkeleton Skeleton { get; set; }

    public float KeypointConfidence { get; set; }

    public float KeypointRadius { get; set; }

    public float KeypointLineWidth { get; set; }

    public PosePlottingOptions()
    {
        Skeleton = new HumanSkeleton();
        KeypointConfidence = .5F;
        KeypointRadius = 3F;
        KeypointLineWidth = 1.5F;
    }
}