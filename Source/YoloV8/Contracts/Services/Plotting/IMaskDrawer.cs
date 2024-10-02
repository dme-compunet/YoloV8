namespace Compunet.YoloV8.Contracts.Services.Plotting;

internal interface IMaskDrawer
{
    public void DrawMask(Segmentation prediction, PlottingContext context);
}