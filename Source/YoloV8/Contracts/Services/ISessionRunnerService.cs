namespace Compunet.YoloV8.Contracts.Services;

internal interface ISessionRunnerService
{
    public IYoloRawOutput PreprocessAndRun(Image<Rgb24> image, out PredictorTimer timer);
}