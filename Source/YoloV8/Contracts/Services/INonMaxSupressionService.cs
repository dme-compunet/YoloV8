namespace Compunet.YoloV8.Contracts.Services;

internal interface INonMaxSuppressionService
{
    public T[] Suppress<T>(Span<T> boxes, float iouThreshold) where T : IRawBoundingBox<T>;
}