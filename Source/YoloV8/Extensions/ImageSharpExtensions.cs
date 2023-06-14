namespace Compunet.YoloV8.Extensions;

public static class ImageSharpExtensions
{
    public static void ForEachPixel<TPixel>(this Image<TPixel> image, Action<Point, TPixel> action)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        var width = image.Width;
        var height = image.Height;

        Parallel.For(0, width, x =>
        {
            Parallel.For(0, height, y =>
            {
                var point = new Point(x, y);
                var pixel = image[x, y];

                action(point, pixel);
            });
        });
    }
}