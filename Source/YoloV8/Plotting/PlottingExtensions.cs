using Compunet.YoloV8.Data;

namespace Compunet.YoloV8.Plotting;

public static class PlottingExtensions
{
    public static Image PlotImage(this IPoseResult result, Image origin) => PlottingUtilities.PlotImage(origin, result);

    public static Image PlotImage(this IPoseResult result, Image origin, ISkeleton skeleton) => PlottingUtilities.PlotImage(origin, result, skeleton);

    public static Image PlotImage(this IDetectionResult result, Image origin) => PlottingUtilities.PlotImage(origin, result);
}
