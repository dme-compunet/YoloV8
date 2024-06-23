namespace Compunet.YoloV8;

public class YoloV8Builder : IYoloV8Builder
{
    private BinarySelector? _model;

    private SessionOptions? _sessionOptions;

    private YoloV8Metadata? _metadata;
    private YoloV8Configuration? _configuration;

    public static IYoloV8Builder CreateDefaultBuilder()
    {
        var builder = new YoloV8Builder();

#if GPURELEASE
        builder.UseCuda(0);
#endif

        return builder;
    }

    public YoloV8Predictor Build()
    {
        if (_model is null)
        {
            throw new ApplicationException("No model selected");
        }

        return new YoloV8Predictor(_model, _metadata, _configuration, _sessionOptions);
    }

    public IYoloV8Builder UseOnnxModel(BinarySelector model)
    {
        _model = model;

        return this;
    }

#if GPURELEASE

   public IYoloV8Builder UseCuda(int deviceId) => WithSessionOptions(SessionOptions.MakeSessionOptionWithCudaProvider(deviceId));
   public IYoloV8Builder UseCuda(OrtCUDAProviderOptions options) => WithSessionOptions(SessionOptions.MakeSessionOptionWithCudaProvider(options));

   public IYoloV8Builder UseRocm(int deviceId) => WithSessionOptions(SessionOptions.MakeSessionOptionWithRocmProvider(deviceId));
   public IYoloV8Builder UseRocm(OrtROCMProviderOptions options) => WithSessionOptions(SessionOptions.MakeSessionOptionWithRocmProvider(options));

   public IYoloV8Builder UseTensorrt(int deviceId) => WithSessionOptions(SessionOptions.MakeSessionOptionWithTensorrtProvider(deviceId));
   public IYoloV8Builder UseTensorrt(OrtTensorRTProviderOptions options) => WithSessionOptions(SessionOptions.MakeSessionOptionWithTensorrtProvider(options));

   public IYoloV8Builder UseTvm(string settings = "") => WithSessionOptions(SessionOptions.MakeSessionOptionWithTvmProvider(settings));

#endif

    public IYoloV8Builder WithMetadata(YoloV8Metadata metadata)
    {
        _metadata = metadata;

        return this;
    }

    public IYoloV8Builder WithConfiguration(Action<YoloV8Configuration> configure)
    {
        var configuration = new YoloV8Configuration();

        configure(configuration);

        _configuration = configuration;

        return this;
    }

    public IYoloV8Builder WithSessionOptions(SessionOptions sessionOptions)
    {
        _sessionOptions = sessionOptions;

        return this;
    }
}