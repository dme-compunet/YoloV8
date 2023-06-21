using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Compunet.YoloV8.Data;
using Compunet.YoloV8.Timing;
using Compunet.YoloV8.Parsers;
using Compunet.YoloV8.Metadata;
using Compunet.YoloV8.Extensions;

namespace Compunet.YoloV8;

public class YoloV8 : IDisposable
{
    #region Private Memebers

    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    private readonly PoseOutputParser _poseParser;
    private readonly DetectionOutputParser _detectionParser;

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
    {
        _inference = new(selector.Load());
        _inputNames = _inference.InputMetadata.Keys.ToArray();

        _metadata = YoloV8Metadata.Parse(_inference.ModelMetadata.CustomMetadataMap);
        _parameters = YoloV8Parameters.Default;

        _detectionParser = new DetectionOutputParser(_metadata, _parameters);
        _poseParser = new PoseOutputParser(_metadata, _parameters);
    }

    public YoloV8(ModelSelector selector, YoloV8Metadata metadata)
    {
        _inference = new(selector.Load());
        _inputNames = _inference.InputMetadata.Keys.ToArray();

        _metadata = metadata;
        _parameters = YoloV8Parameters.Default;

        _detectionParser = new DetectionOutputParser(_metadata, _parameters);
        _poseParser = new PoseOutputParser(_metadata, _parameters);
    }

    #endregion

    #region Detect

    public IDetectionResult Detect(ImageSelector selector)
    {
        EnsureTask(YoloV8Task.Detect);

        return RunPreprocessAndInference(selector, (outputs, image, timer) =>
        {
            var output = outputs[0].AsTensor<float>();

            var boxes = _detectionParser.Parse(output, image);

            var speed = timer.Stop();

            return new DetectionResult(image,
                                       speed,
                                       boxes);
        });
    }

    #endregion

    #region Pose

    public IPoseResult Pose(ImageSelector selector)
    {
        EnsureTask(YoloV8Task.Pose);

        return RunPreprocessAndInference(selector, (outputs, image, timer) =>
        {
            var output = outputs[0].AsTensor<float>();

            var poses = _poseParser.Parse(output, image);

            var speed = timer.Stop();

            return new PoseResult(image,
                                  speed,
                                  poses);
        });
    }

    #endregion

    #region Classify

    public IClassificationResult Classify(ImageSelector selector)
    {
        EnsureTask(YoloV8Task.Classify);

        return RunPreprocessAndInference<IClassificationResult>(selector, (outputs, image, timer) =>
        {
            var output = outputs[0].AsEnumerable<float>().ToList();

            var probs = new List<(YoloV8Class Class, float Confidence)>(output.Count);

            for (int i = 0; i < output.Count; i++)
            {
                var cls = _metadata.Classes[i];
                var scr = output[i];

                var prob = (Class: cls, Confidence: scr);

                probs.Add(prob);
            }

            var speed = timer.Stop();

            return new ClassificationResult(image, speed, probs);
        });
    }

    #endregion

    #region Private Methods

    private TResult RunPreprocessAndInference<TResult>(ImageSelector selector,
                                                       PostprocessContext<TResult> postprocess)
    {
        using var image = selector.Load();

        image.Mutate(x => x.AutoOrient());

        var origin = image.Size;

        var timer = new SpeedTimer();

        timer.StartPreprocess();

        var input = Preprocess(image);

        var inputs = MapToNames(new Tensor<float>[] { input });

        timer.StartInference();

        using var outputs = _inference.Run(inputs);

        var list = new List<NamedOnnxValue>(outputs);

        timer.StartPostprocess();

        return postprocess(list, origin, timer);
    }

    private Tensor<float> Preprocess(Image<Rgb24> image)
    {
        var size = _metadata.ImageSize;

        image.Mutate(x => x.Resize(size));

        var dimensions = new int[] { 1, 3, size.Height, size.Width };
        var input = new DenseTensor<float>(dimensions);

        image.ForEachPixel((point, pixel) =>
        {
            int x = point.X;
            int y = point.Y;

            float r = pixel.R / 255f;
            float g = pixel.G / 255f;
            float b = pixel.B / 255f;

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
        }).ToArray();
    }

    private void EnsureTask(YoloV8Task task)
    {
        if (_metadata.Task != task)
            throw new InvalidOperationException("The loaded model does not support this task");
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