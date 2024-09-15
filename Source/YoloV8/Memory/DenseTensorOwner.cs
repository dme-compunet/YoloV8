namespace Compunet.YoloV8.Memory;

internal class DenseTensorOwner<T>(IMemoryOwner<T> owner, ReadOnlySpan<int> dimensions) : IDisposable
{
    private DenseTensor<T>? _tensor = new(owner.Memory, dimensions);

    public DenseTensor<T> Tensor
    {
        get
        {
            ObjectDisposedException.ThrowIf(_tensor is null, this);
            return _tensor;
        }
    }

    public void Dispose()
    {
        if (_tensor != null)
        {
            _tensor = null;
            owner.Dispose();
        }
    }
}