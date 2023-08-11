namespace Compunet.YoloV8.Extensions;

public static class ImageSharpExtensions
{
    public static void ForEachPixel<TPixel>(this Image<TPixel> image, Action<Point, TPixel> action)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        var width = image.Width;
        var height = image.Height;

        var flag = image.DangerousTryGetSinglePixelMemory(out Memory<TPixel> memory);

        if (flag)
        {
            var totalPixels = width * height;
            Parallel.For(0, totalPixels, index =>
            {
                int x = index % width;
                int y = index / width;

                var point = new Point(x, y);
                var pixel = memory.Span[index];

                action(point, pixel);
            });
        }
        else
        {
            Parallel.For(0, width, x =>
            {
                for (int y = 0; y < height; y++)
                {
                    var point = new Point(x, y);
                    var pixel = image[x, y];

                    action(point, pixel);
                }
            });
        }
    }
}