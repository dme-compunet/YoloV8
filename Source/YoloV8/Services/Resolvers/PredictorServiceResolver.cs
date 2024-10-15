namespace Compunet.YoloV8.Services.Resolvers;

internal class PredictorServiceResolver : IDisposable
{
    private readonly YoloSession _yoloSession;
    private readonly YoloConfiguration _configuration;

    private readonly ServiceProvider _provider;
    private readonly Dictionary<YoloConfiguration, ServiceProvider> _providers = [];

    private bool _disposed;

    public PredictorServiceResolver(InferenceSession session, YoloConfiguration configuration)
    {
        _configuration = configuration;

        var metadata = YoloMetadata.Parse(session);
        var shapeInfo = new SessionIoShapeInfo(session, metadata);

        // Create default services
        var services = CreateDefaultServices(metadata);

        // Create yolo session
        _yoloSession = new YoloSession(metadata, session, shapeInfo);

        // Add yolo session
        services.AddSingleton(_yoloSession);
        services.AddSingleton(_configuration);

        // Build the service provider
        _provider = services.BuildServiceProvider();
    }

    public T Resolve<T>(YoloConfiguration? configuration = null) where T : notnull
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (configuration is null || _configuration.Equals(configuration))
        {
            return _provider.GetRequiredService<T>();
        }

        if (_providers.TryGetValue(configuration, out var p))
        {
            return p.GetRequiredService<T>();
        }
        else
        {
            var services = CreateDefaultServices(_yoloSession.Metadata);

            services.AddSingleton(_yoloSession);
            services.AddSingleton(configuration);

            var provider = services.BuildServiceProvider();

            _providers.Add(configuration, provider);

            return provider.GetRequiredService<T>();
        }
    }

    private static ServiceCollection CreateDefaultServices(YoloMetadata metadata)
    {
        var services = new ServiceCollection();

        if (metadata is YoloPoseMetadata pose)
        {
            services.AddSingleton(pose);
        }

        services
            .AddSingleton(metadata)
            .AddSingleton<IPixelsNormalizerService, PixelsNormalizerService>()
            .AddSingleton<ISessionRunnerService, SessionRunnerService>()
            .AddSingleton<IRawBoundingBoxParser, RawBoundingBoxParser>()
            .AddSingleton<IMemoryAllocatorService, MemoryAllocatorService>()
            .AddSingleton<IImageAdjustmentService, ImageAdjustmentService>()
            .AddSingleton<INonMaxSuppressionService, NonMaxSuppressionService>();

        switch (metadata.Task)
        {
            case YoloTask.Pose:
                services.AddSingleton<IParser<Pose>, PoseParser>();
                break;

            case YoloTask.Detect:
                services.AddSingleton<IParser<Detection>, DetectionParser>();
                break;

            case YoloTask.Obb:
                services.AddSingleton<IParser<ObbDetection>, ObbDetectionParser>();
                break;

            case YoloTask.Segment:
                services.AddSingleton<IParser<Segmentation>, SegmentationParser>();
                break;

            case YoloTask.Classify:
                services.AddSingleton<IParser<Classification>, ClassificationParser>();
                break;
        }

        return services;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _provider.Dispose();

        foreach (var provider in _providers.Values)
        {
            provider.Dispose();
        }

        _disposed = true;
    }
}