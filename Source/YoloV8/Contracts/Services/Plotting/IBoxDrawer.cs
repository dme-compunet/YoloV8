namespace Compunet.YoloV8.Contracts.Services.Plotting;

internal interface IBoxDrawer
{
    public void DrawBox(YoloPrediction prediction, PlottingContext context);
}