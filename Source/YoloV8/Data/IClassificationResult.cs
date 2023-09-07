namespace Compunet.YoloV8.Data;

public interface IClassificationResult : IYoloV8Result
{
    YoloV8Class Class { get; }

    float Confidence { get; }

    IReadOnlyList<(YoloV8Class Class, float Confidence)> Probabilities { get; }
}