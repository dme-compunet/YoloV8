namespace Compunet.YoloV8;

public class YoloV8Predictor : IDisposable
{
    #region Private Memebers

    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    private readonly InferenceSession _inference;
    private readonly string[] _inputNames;

    private readonly object _sync = new();

    private bool _disposed;

    #endregion

    #region Public Properties

    public YoloV8Metadata Metadata => _metadata;

    public YoloV8Parameters Parameters => _parameters;

    #endregion

    #region Ctors

    public YoloV8Predictor(ModelSelector selector)
        : this(selector.Load(), null, null)
    { }

    public YoloV8Predictor(ModelSelector selector, SessionOptions options)
        : this(selector.Load(), null, options)
    { }

    public YoloV8Predictor(ModelSelector selector, YoloV8Metadata metadata)
        : this(selector.Load(), metadata, null)
    { }

    public YoloV8Predictor(ModelSelector selector, YoloV8Metadata metadata, SessionOptions options)
        : this(selector.Load(), metadata, options)
    { }

    private YoloV8Predictor(byte[] model, YoloV8Metadata? metadata, SessionOptions? options)
    {
        _inference = new(model, options ?? new SessionOptions());
        _inputNames = _inference.InputMetadata.Keys.ToArray();

        _metadata = metadata ?? YoloV8Metadata.Parse(_inference.ModelMetadata.CustomMetadataMap);
        _parameters = YoloV8Parameters.Default;
    }

    #endregion

    #region Run

    public TResult Run<TResult>(ImageSelector selector, PostprocessContext<TResult> postprocess) where TResult : YoloV8Result
    {
        using var image = selector.Load(true);

        var originSize = image.Size;

        var timer = new SpeedTimer();

        timer.StartPreprocess();

        var input = Preprocess(image);

        var inputs = MapNamedOnnxValues([input]);

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
        if (_parameters.SuppressParallelInference)
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

        PreprocessHelper.ProcessToTensor(image, modelSize, _parameters.KeepOriginalAspectRatio, input, 0);

        return input;
    }

    private NamedOnnxValue[] MapNamedOnnxValues(ReadOnlySpan<Tensor<float>> inputs)
    {
        var length = inputs.Length;

        var values = new NamedOnnxValue[length];

        for (int i = 0; i < length; i++)
        {
            var name = _inputNames[i];

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
            return;

        _inference.Dispose();
        _disposed = true;

        GC.SuppressFinalize(this);
    }

    #endregion
}