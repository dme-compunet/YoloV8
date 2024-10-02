namespace Compunet.YoloV8.Contracts.Services.Plotting;

internal interface IPlotter<T> where T : IYoloPrediction<T>
{
    public void Plot(YoloResult<T> result, PlottingContext context);
}