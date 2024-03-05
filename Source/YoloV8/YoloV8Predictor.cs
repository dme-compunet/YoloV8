namespace Compunet.YoloV8;

public class YoloV8Predictor : IDisposable
{
    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Configuration _configuration;

    private readonly InferenceSession _inference;

    private readonly object _locker = new();

    private bool _disposed;

    public YoloV8Metadata Metadata => _metadata;

    public YoloV8Configuration Configuration => _configuration;

    public static YoloV8Predictor Create(BinarySelector model) => YoloV8Builder.CreateDefaultBuilder().UseOnnxModel(model).Build();

    internal YoloV8Predictor(BinarySelector model, YoloV8Metadata? metadata, YoloV8Configuration? configuration, SessionOptions? options)
    {
        _inference = new InferenceSession(model.Load(), options ?? new SessionOptions());

        _metadata = metadata ?? YoloV8Metadata.Parse(_inference.ModelMetadata.CustomMetadataMap);
        _configuration = configuration ?? YoloV8Configuration.Default;
    }

    public TResult Run<TResult>(ImageSelector selector, PostprocessContext<TResult> postprocess) where TResult : YoloV8Result
    {
        using var image = selector.Load(true);

        var originSize = image.Size;

        var timer = new SpeedTimer();

        timer.StartPreprocess();

        var input = Preprocess(image);

        var inputs = CreateInputAndMapNames([input]);

        timer.StartInference();

        using var outputs = Infer(inputs);

        var list = new List<NamedOnnxValue>(outputs);

        timer.StartPostprocess();

        return postprocess(list, originSize, timer);
    }

    private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Infer(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        if (_configuration.SuppressParallelInference)
        {
            lock (_locker)
            {
                return _inference.Run(inputs);
            }
        }

        return _inference.Run(inputs);
    }

    private Tensor<float> Preprocess(Image<Rgb24> image)
    {
        var modelSize = _metadata.ImageSize;

        var dimensions = new int[] { 1, 3, modelSize.Height, modelSize.Width };
        var input = new DenseTensor<float>(dimensions);

        PreprocessHelper.ProcessToTensor(image, modelSize, _configuration.KeepOriginalAspectRatio, input, 0);

        return input;
    }

    private NamedOnnxValue[] CreateInputAndMapNames(ReadOnlySpan<Tensor<float>> inputs)
    {
        var length = inputs.Length;

        var values = new NamedOnnxValue[length];

        for (int i = 0; i < length; i++)
        {
            var name = _inference.InputNames[i];

            var value = NamedOnnxValue.CreateFromTensor(name, inputs[i]);

            values[i] = value;
        }

        return values;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _inference.Dispose();
        _disposed = true;

        GC.SuppressFinalize(this);
    }
}