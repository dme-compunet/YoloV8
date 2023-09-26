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

        Parallel.For(0, image.Height, y =>
        {
            var row = image.DangerousGetPixelRowMemory(y).Span;

            for (int x = 0; x < image.Width; x++)
            {
                var pixel = row[x];

                target[batch, 0, y + yPadding, x + xPadding] = pixel.R / 255f;
                target[batch, 1, y + yPadding, x + xPadding] = pixel.G / 255f;
                target[batch, 2, y + yPadding, x + xPadding] = pixel.B / 255f;
            }
        });
    }
}