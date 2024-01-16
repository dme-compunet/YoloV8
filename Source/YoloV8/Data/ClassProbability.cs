namespace Compunet.YoloV8.Data;

public class ClassProbability
{
    public required YoloV8Class Class { get; init; }

    public required float Confidence { get; init; }

    public override string ToString()
    {
        return $"{Class.Name} ({Confidence:N})";
    }
}