namespace Compunet.YoloV8.Extensions;

public static class ImageSharpExtensions
{
    public static void ForEachPixel<TPixel>(this Image<TPixel> image, Action<Point, TPixel> action)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        if (ForEachPixelOptimized(image, action))
            return;

        var width = image.Width;
        var height = image.Height;

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

    private static bool ForEachPixelOptimized<TPixel>(this Image<TPixel> image, Action<Point, TPixel> action)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        var succes = image.DangerousTryGetSinglePixelMemory(out Memory<TPixel> memory);

        if (succes == false)
            return false;

        var width = image.Width;
        var height = image.Height;
        var totalPixels = width * height;

        Parallel.For(0, totalPixels, index =>
        {
            int x = index % width;
            int y = index / width;

            var point = new Point(x, y);
            var pixel = memory.Span[index];

            action(point, pixel);
        });

        return true;
    }
}