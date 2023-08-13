using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Compunet.YoloV8.Timing;
using Compunet.YoloV8.Metadata;
using Compunet.YoloV8.Extensions;

namespace Compunet.YoloV8;

public class YoloV8 : IDisposable
{
    #region Private Memebers

    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    private readonly InferenceSession _inference;
    private readonly string[] _inputNames;

    private bool _disposed;

    #endregion

    #region Public Properties

    public YoloV8Metadata Metadata => _metadata;

    public YoloV8Parameters Parameters => _parameters;

    #endregion

    #region Ctors

    public YoloV8(ModelSelector selector)
        : this(selector.Load(), null, null)
    { }

    public YoloV8(ModelSelector selector, SessionOptions options)
        : this(selector.Load(), null, options)
    { }

    public YoloV8(ModelSelector selector, YoloV8Metadata metadata)
        : this(selector.Load(), metadata, null)
    { }

    public YoloV8(ModelSelector selector, YoloV8Metadata metadata, SessionOptions options)
        : this(selector.Load(), metadata, options)
    { }

    private YoloV8(byte[] model, YoloV8Metadata? metadata, SessionOptions? options)
    {
        _inference = new(model, options ?? new SessionOptions());
        _inputNames = _inference.InputMetadata.Keys.ToArray();

        _metadata = metadata ?? YoloV8Metadata.Parse(_inference.ModelMetadata.CustomMetadataMap);
        _parameters = YoloV8Parameters.Default;
    }

    #endregion

    #region Run

    public TResult Run<TResult>(ImageSelector selector, PostprocessContext<TResult> postprocess)
    {
        using var image = selector.Load();

        image.Mutate(x => x.AutoOrient());

        var originSize = image.Size;

        var timer = new SpeedTimer();

        timer.StartPreprocess();

        var input = Preprocess(image);

        var inputs = MapToNames(new Tensor<float>[] { input });

        timer.StartInference();

        using var outputs = _inference.Run(inputs);

        var list = new List<NamedOnnxValue>(outputs);

        timer.StartPostprocess();

        return postprocess(list, originSize, timer);
    }

    #endregion

    #region Private Methods

    private Tensor<float> Preprocess(Image<Rgb24> image)
    {
        var modelSize = _metadata.ImageSize;

        var ratio = Math.Min((float)modelSize.Width / image.Width, (float)modelSize.Height / image.Height);

        var processSize = new Size((int)(image.Width * ratio), (int)(image.Height * ratio));

        var xPadding = (modelSize.Width - processSize.Width) / 2;
        var yPadding = (modelSize.Height - processSize.Height) / 2;

        image.Mutate(x => x.Resize(processSize));

        var dimensions = new int[] { 1, 3, modelSize.Height, modelSize.Width };
        var input = new DenseTensor<float>(dimensions);

        image.ForEachPixel((point, pixel) =>
        {
            var x = point.X + xPadding;
            var y = point.Y + yPadding;

            var r = pixel.R / 255f;
            var g = pixel.G / 255f;
            var b = pixel.B / 255f;

            input[0, 0, y, x] = r;
            input[0, 1, y, x] = g;
            input[0, 2, y, x] = b;
        });

        return input;
    }

    private IReadOnlyList<NamedOnnxValue> MapToNames(IReadOnlyList<Tensor<float>> inputs)
    {
        return inputs.Select((value, index) =>
        {
            var name = _inputNames[index];
            return NamedOnnxValue.CreateFromTensor(name, value);
        }
        ).ToArray();
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