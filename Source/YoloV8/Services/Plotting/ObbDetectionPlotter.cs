namespace Compunet.YoloV8.Services.Plotting;

internal class ObbDetectionPlotter(IBoxDrawer boxPlotter,
                                   INameDrawer namePlotter) : IPlotter<ObbDetection>
{
    public void Plot(YoloResult<ObbDetection> result, PlottingContext context)
    {
        foreach (var box in result)
        {
            boxPlotter.DrawBox(box, context);
            namePlotter.DrawName(box, box.Bounds.Location, false, context);
        }
    }
}