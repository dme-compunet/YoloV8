namespace Compunet.YoloV8.Memory;

internal interface IYoloRawOutput : IDisposable
{
    public DenseTensor<float> Output0 { get; }

    public DenseTensor<float>? Output1 { get; }
}
