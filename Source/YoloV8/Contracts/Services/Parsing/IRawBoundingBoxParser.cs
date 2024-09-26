namespace Compunet.YoloV8.Contracts.Services;

internal interface IRawBoundingBoxParser
{
    public T[] Parse<T>(DenseTensor<float> tensor) where T : IRawBoundingBox<T>;
}