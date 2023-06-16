using Compunet.YoloV8.Timing;
using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Data;

internal class ClassificationResult : YoloV8Result, IClassificationResult
{
    public YoloV8Class Class { get; }

    public float Confidence { get; }

    public IReadOnlyList<(YoloV8Class Class, float Confidence)> Probabilities { get; }

    public ClassificationResult(Size image,
                                SpeedResult speed,
                                IReadOnlyList<(YoloV8Class Class, float Confidence)> probabilities)
        : base(image, speed)
    {
        Probabilities = probabilities;

        var top = Probabilities.MaxBy(x => x.Confidence);

        Class = top.Class;
        Confidence = top.Confidence;
    }

    public override string ToString()
    {
        return $"{Class.Name} {Confidence:N}";
    }
}