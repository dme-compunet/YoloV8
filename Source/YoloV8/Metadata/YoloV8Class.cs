namespace Compunet.YoloV8.Metadata;

public class YoloV8Class(int id, string name)
{
    public int Id { get; } = id;

    public string Name { get; } = name;

    public override string ToString()
    {
        return $"{Id}: '{Name}'";
    }
}