namespace Compunet.YoloV8.Contracts.Services.Plotting;

internal interface INameDrawer
{
    public void DrawName(YoloPrediction prediction, PointF position, bool inside, PlottingContext context);
}