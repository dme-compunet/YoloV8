namespace Compunet.YoloV8.Memory;

internal class OrtYoloRawOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result) : IYoloRawOutput
{
    public DenseTensor<float> Output0 => ForceDenseTensor(result[0]);

    public DenseTensor<float>? Output1
    {
        get
        {
            if (result.Count > 1)
            {
                return ForceDenseTensor(result[1]);
            }

            return null;
        }
    }

    public void Dispose() => result.Dispose();

    private static DenseTensor<float> ForceDenseTensor(NamedOnnxValue value)
    {
        return value.AsTensor<float>() as DenseTensor<float>
               ?? 
               throw new InvalidOperationException("The ort result is not DenseTensor");
    }
}

