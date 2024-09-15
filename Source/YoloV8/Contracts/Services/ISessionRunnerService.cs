namespace Compunet.YoloV8.Contracts.Services;

internal interface ISessionRunnerService
{
    public YoloRawOutput PreprocessAndRun(Image<Rgb24> image, out PredictorTimer timer);
}