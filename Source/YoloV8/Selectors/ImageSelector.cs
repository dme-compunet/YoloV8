namespace Compunet.YoloV8.Selectors;

public class ImageSelector
{
    private readonly Func<Image<Rgb24>> _factory;

    public ImageSelector(string path)
    {
        _factory = () => Image.Load<Rgb24>(path);
    }

    public ImageSelector(byte[] data)
    {
        _factory = () => Image.Load<Rgb24>(data);
    }

    internal Image<Rgb24> Load() => _factory();

    public static implicit operator ImageSelector(string path) => new(path);
    public static implicit operator ImageSelector(byte[] data) => new(data);
}