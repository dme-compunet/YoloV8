namespace Compunet.YoloSharp.Contracts.Services;

internal interface IRawBoundingBoxParser
{
    public T[] Parse<T>(MemoryTensor<float> tensor) where T : IRawBoundingBox<T>;
}