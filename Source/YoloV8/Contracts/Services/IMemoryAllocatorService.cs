namespace Compunet.YoloV8.Contracts.Services;

internal interface IMemoryAllocatorService
{
    public IMemoryOwner<T> Allocate<T>(int length, bool clean = false);
}