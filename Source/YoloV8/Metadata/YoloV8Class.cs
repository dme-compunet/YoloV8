namespace Compunet.YoloV8.Metadata;

public class YoloV8Class
{
    public int Id { get; }

    public string Name { get; }

    public YoloV8Class(int id, string name)
    {
        Id = id;
        Name = name;
    }

    public override string ToString()
    {
        return $"{Id}: '{Name}'";
    }
}