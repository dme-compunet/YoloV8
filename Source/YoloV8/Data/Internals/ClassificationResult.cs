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
        var top = probabilities.MaxBy(x => x.Confidence);

        Class = top.Class;
        Confidence = top.Confidence;

        Probabilities = probabilities;
    }

    public override string ToString()
    {
        return $"{Class.Name} ({Confidence:N})";
    }
}