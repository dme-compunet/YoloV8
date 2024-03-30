namespace Compunet.YoloV8.Plotting;

public static class PlottingAsyncOperationExtensions
{
    public static async Task<Image> PlotImageAsync(this PoseResult result, Image originImage, PosePlottingOptions? options = null)
    {
        return await Task.Run(() => result.PlotImage(originImage, options));
    }

    public static async Task<Image> PlotImageAsync(this DetectionResult result, Image originImage, DetectionPlottingOptions? options = null)
    {
        return await Task.Run(() => result.PlotImage(originImage, options));
    }

    public static async Task<Image> PlotImageAsync(this ObbDetectionResult result, Image originImage, DetectionPlottingOptions? options = null)
    {
        return await Task.Run(() => result.PlotImage(originImage, options));
    }

    public static async Task<Image> PlotImageAsync(this SegmentationResult result, Image originImage, SegmentationPlottingOptions? options = null)
    {
        return await Task.Run(() => result.PlotImage(originImage, options));
    }


    public static async Task<Image> PlotImageAsync(this ClassificationResult result, Image originImage, ClassificationPlottingOptions? options = null)
    {
        return await Task.Run(() => result.PlotImage(originImage, options));
    }
}