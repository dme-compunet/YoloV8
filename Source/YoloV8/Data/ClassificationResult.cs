namespace Compunet.YoloV8.Data;

public class ClassificationResult : YoloV8Result
{
    public required ClassProbability TopClass { get; init; }

    public required IEnumerable<ClassProbability> Probabilities { get; init; }

    public override string ToString()
    {
        return TopClass.ToString();
    }
}