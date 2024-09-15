namespace Compunet.YoloV8.Memory;

internal class YoloRawOutput(DenseTensorOwner<float> output0, DenseTensorOwner<float>? output1) : IDisposable
{
    private bool _disposed;

    public DenseTensor<float> Output0
    {
        get
        {
            EnsureNotDisposed();
            return output0.Tensor;
        }
    }

    public DenseTensor<float>? Output1
    {
        get
        {
            EnsureNotDisposed();
            return output1?.Tensor;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        output0.Dispose();
        output1?.Dispose();

        _disposed = true;
    }

    private void EnsureNotDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}