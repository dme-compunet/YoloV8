namespace Compunet.YoloV8.Memory;

internal static class MemoryAllocatorExtensions
{
    public static DenseTensorOwner<T> AllocateTensor<T>(this IMemoryAllocatorService allocator, TensorShape shape, bool clean = false)
    {
        var memory = allocator.Allocate<T>(shape.Length, clean);

        return new DenseTensorOwner<T>(memory, shape.Dimensions);
    }
}