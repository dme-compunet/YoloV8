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

        if (IsGpuNuGetPackage())
        {
            builder.UseCuda(0);
        }

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

    public IYoloV8Builder UseCuda(int deviceId)
    {
        _sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(deviceId);

        return this;
    }

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

    private static bool IsGpuNuGetPackage()
    {
        var assemblyName = Assembly.GetExecutingAssembly().GetName().Name;

        if (assemblyName == "YoloV8.Gpu")
        {
            return true;
        }

        return false;
    }
}