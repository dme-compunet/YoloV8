namespace Compunet.YoloV8.Contracts.Services;

internal interface IParser<T>
{
    public T[] ProcessTensorToResult(YoloRawOutput output, Size size);
}