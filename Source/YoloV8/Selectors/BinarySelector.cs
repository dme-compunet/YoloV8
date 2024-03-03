namespace Compunet.YoloV8;

public class BinarySelector
{
    private readonly Func<byte[]> _factory;

    public BinarySelector(string path)
    {
        _factory = () => File.ReadAllBytes(path);
    }

    public BinarySelector(byte[] data)
    {
        _factory = () => data;
    }

    public BinarySelector(Stream stream)
    {
        _factory = () =>
        {
            using var memory = new MemoryStream();
            stream.CopyTo(memory);

            return memory.ToArray();
        };
    }

    internal byte[] Load() => _factory();

    public static implicit operator BinarySelector(string path) => new(path);
    public static implicit operator BinarySelector(byte[] data) => new(data);
    public static implicit operator BinarySelector(Stream stream) => new(stream);
}