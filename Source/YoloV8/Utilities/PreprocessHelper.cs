namespace Compunet.YoloV8.Utilities;

internal static class PreprocessHelper
{
    public static void ProcessToTensor(Image<Rgb24> image, Size modelSize, bool originalAspectRatio, DenseTensor<float> target, int batch)
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

                WritePixel(batch, y + yPadding, x + xPadding, pixel, target);
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

                    WritePixel(batch, y + yPadding, x + xPadding, pixel, target);
                }
            });
        }
    }

    private static void WritePixel(int batch, int y, int x, Rgb24 pixel, DenseTensor<float> target)
    {
        var offsetR = target.Strides[0] * batch
                    + target.Strides[1] * 0
                    + target.Strides[2] * y
                    + target.Strides[3] * x;

        var offsetG = target.Strides[0] * batch
                    + target.Strides[1] * 1
                    + target.Strides[2] * y
                    + target.Strides[3] * x;

        var offsetB = target.Strides[0] * batch
                    + target.Strides[1] * 2
                    + target.Strides[2] * y
                    + target.Strides[3] * x;

        target.Buffer.Span[offsetR] = pixel.R / 255f;
        target.Buffer.Span[offsetG] = pixel.G / 255f;
        target.Buffer.Span[offsetB] = pixel.B / 255f;
    }
}