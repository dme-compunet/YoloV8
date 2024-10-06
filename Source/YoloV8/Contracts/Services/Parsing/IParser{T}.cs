namespace Compunet.YoloV8.Contracts.Services;

internal interface IParser<T> where T : IYoloPrediction<T>
{
    public T[] ProcessTensorToResult(IYoloRawOutput output, Size size);
}