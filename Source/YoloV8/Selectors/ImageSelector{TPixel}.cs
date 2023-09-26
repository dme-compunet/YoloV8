namespace Compunet.YoloV8;

public class ImageSelector<TPixel> where TPixel : unmanaged, IPixel<TPixel>
{
    private readonly Func<Image<TPixel>> _factory;

    public ImageSelector(Image image)
    {
        _factory = image.CloneAs<TPixel>;
    }

    public ImageSelector(string path)
    {
        _factory = () => Image.Load<TPixel>(path);
    }

    public ImageSelector(byte[] data)
    {
        _factory = () => Image.Load<TPixel>(data);
    }

    public ImageSelector(Stream stream)
    {
        _factory = () => Image.Load<TPixel>(stream);
    }

    internal Image<TPixel> Load(bool autoOrient)
    {
        var image = _factory();

        if (autoOrient)
            image.Mutate(x => x.AutoOrient());

        return image;
    }

    public static implicit operator ImageSelector<TPixel>(Image image) => new(image);
    public static implicit operator ImageSelector<TPixel>(string path) => new(path);
    public static implicit operator ImageSelector<TPixel>(byte[] data) => new(data);
    public static implicit operator ImageSelector<TPixel>(Stream stream) => new(stream);
}