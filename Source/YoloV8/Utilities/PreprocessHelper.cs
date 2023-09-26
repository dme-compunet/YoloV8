namespace Compunet.YoloV8.Utilities;

internal static class PreprocessHelper
{
    public static void ProcessToTensor(Image<Rgb24> image, Size modelSize, bool originalAspectRatio, Tensor<float> target, int batch)
    {
        var options = new ResizeOptions()
        {
            Size = modelSize,
            Mode = originalAspectRatio ? ResizeMode.Max : ResizeMode.Stretch,
        };

        image.Mutate(x => x.Resize(options));

        var xPadding = (modelSize.Width - image.Width) / 2;
        var yPadding = (modelSize.Height - image.Height) / 2;

        var width = image.Width;
        var height = image.Height;

        // Try get continuous memory block of the entire image data
        if (image.DangerousTryGetSinglePixelMemory(out var memory))
        {
            Parallel.For(0, width * height, index =>
            {
                int x = index % width;
                int y = index / width;

                var pixel = memory.Span[index];

                target[batch, 0, y + yPadding, x + xPadding] = pixel.R / 255f;
                target[batch, 1, y + yPadding, x + xPadding] = pixel.G / 255f;
                target[batch, 2, y + yPadding, x + xPadding] = pixel.B / 255f;
            });
        }
        else
        {
            Parallel.For(0, height, y =>
            {
                var row = image.DangerousGetPixelRowMemory(y).Span;

                for (int x = 0; x < width; x++)
                {
                    var pixel = row[x];

                    target[batch, 0, y + yPadding, x + xPadding] = pixel.R / 255f;
                    target[batch, 1, y + yPadding, x + xPadding] = pixel.G / 255f;
                    target[batch, 2, y + yPadding, x + xPadding] = pixel.B / 255f;
                }
            });
        }
    }
}