namespace Compunet.YoloV8.Contracts.Services;

internal interface IRawBoundingBoxParser
{
    public T[] Parse<T>(DenseTensor<float> tensor, Size imageSize) where T : IRawBoundingBox<T>;

    public T[] Parse<T>(DenseTensor<float> tensor, Size imageSize, Vector<int> padding) where T : IRawBoundingBox<T>;

    public T[] Parse<T>(DenseTensor<float> tensor, Vector<int> padding, Vector<float> ratio) where T : IRawBoundingBox<T>;
}