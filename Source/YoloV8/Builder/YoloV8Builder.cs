namespace Compunet.YoloV8;

public class YoloV8Builder : IYoloV8Builder
{
    private readonly BinarySelector _modelSelector;

    private SessionOptions? _sessionOptions;

    private YoloV8Metadata? _metadata;
    private YoloV8Configuration? _configuration;

    private YoloV8Builder(BinarySelector model) => _modelSelector = model;

    public static IYoloV8Builder CreateBuilder(BinarySelector model) => new YoloV8Builder(model);

    public static IYoloV8Builder CreateDefaultBuilder(BinarySelector model)
    {
        var builder = CreateBuilder(model).UseMetadateDetect()
                                          .UseDefaultConfiguration();

        // TODO:
        if (true) // IsGpu
        {
            builder.UseCuda(0);
        }

        return builder;
    }

    public YoloV8Predictor Build()
    {
        var options = _sessionOptions ?? new SessionOptions();
        var configuration = _configuration ?? new YoloV8Configuration();

        return new YoloV8Predictor(_modelSelector, _metadata, configuration, options);
    }

    public IYoloV8Builder UseCuda(int deviceId)
    {
        _sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(deviceId);

        return this;
    }

    public IYoloV8Builder UseDefaultConfiguration()
    {
        _configuration = YoloV8Configuration.Default;

        return this;
    }

    public IYoloV8Builder UseMetadateDetect()
    {
        _metadata = null;

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