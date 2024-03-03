namespace Compunet.YoloV8.Data;

public class ClassProbability
{
    public required YoloV8Class Name { get; init; }

    public required float Confidence { get; init; }

    public override string ToString()
    {
        return $"{Name.Name} ({Confidence:N})";
    }
}