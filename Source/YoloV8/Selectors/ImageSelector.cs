namespace Compunet.YoloV8;

public class ImageSelector
{
    private readonly Func<Image<Rgb24>> _factory;

    public ImageSelector(Image image)
    {
        _factory = () => image.CloneAs<Rgb24>();
    }

    public ImageSelector(string path)
    {
        _factory = () => Image.Load<Rgb24>(path);
    }

    public ImageSelector(byte[] data)
    {
        _factory = () => Image.Load<Rgb24>(data);
    }

    public ImageSelector(Stream stream)
    {
        _factory = () => Image.Load<Rgb24>(stream);
    }

    internal Image<Rgb24> Load() => _factory();

    public static implicit operator ImageSelector(Image image) => new(image);
    public static implicit operator ImageSelector(string path) => new(path);
    public static implicit operator ImageSelector(byte[] data) => new(data);
    public static implicit operator ImageSelector(Stream stream) => new(stream);
}