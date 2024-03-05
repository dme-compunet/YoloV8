namespace Compunet.YoloV8;

public class YoloV8Predictor : IDisposable
{
    #region Private Memebers

    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Configuration _configuration;

    private readonly InferenceSession _inference;

    private readonly object _sync = new();

    private bool _disposed;

    #endregion

    #region Public Properties

    public YoloV8Metadata Metadata => _metadata;

    public YoloV8Configuration Configuration => _configuration;

    #endregion

    #region Ctors

    public static YoloV8Predictor Create(BinarySelector model) => YoloV8Builder.CreateDefaultBuilder().UseOnnxModel(model).Build();

    internal YoloV8Predictor(BinarySelector model, YoloV8Metadata? metadata, YoloV8Configuration? configuration, SessionOptions? options)
    {
        _inference = new InferenceSession(model.Load(), options ?? new SessionOptions());

        _metadata = metadata ?? YoloV8Metadata.Parse(_inference.ModelMetadata.CustomMetadataMap);
        _configuration = configuration ?? YoloV8Configuration.Default;
    }

    //internal YoloV8Predictor(BinarySelector model, YoloV8Metadata? metadata, YoloV8Configuration configuration, SessionOptions options)
    //{
    //    _inference = new InferenceSession(model.Load(), options);
    //    _inputNames = _inference.InputMetadata.Keys.ToArray();

    //    _metadata = metadata ?? YoloV8Metadata.Parse(_inference.ModelMetadata.CustomMetadataMap);

    //    _configuration = configuration;
    //}


    //public YoloV8Predictor(BinarySelector model)
    //    : this(model.Load(), null, null)
    //{ }

    //public YoloV8Predictor(BinarySelector model, SessionOptions options)
    //    : this(model.Load(), null, options)
    //{ }

    //public YoloV8Predictor(BinarySelector model, YoloV8Metadata metadata)
    //    : this(model.Load(), metadata, null)
    //{ }

    //public YoloV8Predictor(BinarySelector model, YoloV8Metadata metadata, SessionOptions options)
    //    : this(model.Load(), metadata, options)
    //{ }

    //private YoloV8Predictor(byte[] model, YoloV8Metadata? metadata, SessionOptions? options)
    //{
    //    _inference = new(model, options ?? new SessionOptions());
    //    _inputNames = _inference.InputMetadata.Keys.ToArray();

    //    _metadata = metadata ?? YoloV8Metadata.Parse(_inference.ModelMetadata.CustomMetadataMap);
    //    _configuration = YoloV8Configuration.Default;
    //}

    #endregion

    #region Run

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

    #endregion

    #region Private Methods

    private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Infer(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        if (_configuration.SuppressParallelInference)
        {
            lock (_sync)
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

    #endregion

    #region Dispose

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

    #endregion
}