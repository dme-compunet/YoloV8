namespace Compunet.YoloV8;

public class YoloPredictorOptions
{
    public static YoloPredictorOptions Default { get; } = new();

#if GPURELEASE
    public bool UseCuda { get; init; } = true;
#else 
    public bool UseCuda { get; init; }
#endif
    public int CudaDeviceId { get; init; }

    public SessionOptions? SessionOptions { get; init; }

    public YoloConfiguration? Configuration { get; init; }

    internal InferenceSession CreateSession(byte[] model)
    {
        if (UseCuda)
        {
            if (SessionOptions is not null)
            {
                throw new InvalidOperationException("'UseCuda' and 'SessionOptions' cannot be used together");
            }

            return new InferenceSession(model, SessionOptions.MakeSessionOptionWithCudaProvider(CudaDeviceId));
        }

        if (SessionOptions != null)
        {
            return new InferenceSession(model, SessionOptions);
        }

        return new InferenceSession(model);
    }
}
