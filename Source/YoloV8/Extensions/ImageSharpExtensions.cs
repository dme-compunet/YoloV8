namespace Compunet.YoloV8.Extensions;

public static class ImageSharpExtensions
{
    public static void ForEachPixel<TPixel>(this Image<TPixel> image, Action<Point, TPixel> action)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        var width = image.Width;
        var height = image.Height;
        var totalPixels = width * height;

        var flag = image.DangerousTryGetSinglePixelMemory(out Memory<TPixel> memory);

        if (flag)
        {
            Parallel.For(0, totalPixels, index =>
            {
                int x = index % width;
                int y = index / width;

                var point = new Point(x, y);
                var pixel = memory.Span[index];

                action(point, pixel);
            });
        }
    }
}