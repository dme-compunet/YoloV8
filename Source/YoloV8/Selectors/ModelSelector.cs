namespace Compunet.YoloV8;

public class ModelSelector
{
    private readonly Func<byte[]> _factory;

    public ModelSelector(string path)
    {
        _factory = () => File.ReadAllBytes(path);
    }

    public ModelSelector(byte[] data)
    {
        _factory = () => data;
    }

    internal byte[] Load() => _factory();

    public static implicit operator ModelSelector(string path) => new(path);
    public static implicit operator ModelSelector(byte[] data) => new(data);
}