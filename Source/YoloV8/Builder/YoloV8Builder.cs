namespace Compunet.YoloV8;

public class YoloV8Builder : IYoloV8Builder
{
    private BinarySelector? _model;

#if GPURELEASE
    private SessionOptions? _sessionOptions;
#endif

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

#if GPURELEASE
        return new YoloV8Predictor(_model, _metadata, _configuration, _sessionOptions);
#else
        return new YoloV8Predictor(_model, _metadata, _configuration, null);
#endif
    }

    public IYoloV8Builder UseOnnxModel(BinarySelector model)
    {
        _model = model;

        return this;
    }

#if GPURELEASE

    public IYoloV8Builder UseCuda(int deviceId)
    {
        _sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(deviceId);

        return this;
    }

    public IYoloV8Builder UseRocm(int deviceId)
    {
        _sessionOptions = SessionOptions.MakeSessionOptionWithRocmProvider(deviceId);

        return this;
    }

    public IYoloV8Builder UseTensorrt(int deviceId)
    {
        _sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(deviceId);

        return this;
    }

    public IYoloV8Builder UseTvm(string settings = "")
    {
        _sessionOptions = SessionOptions.MakeSessionOptionWithTvmProvider(settings);

        return this;
    }

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
}