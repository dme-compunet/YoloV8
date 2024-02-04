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

        // Pre-calculate strides for performance
        var strideBatchR = target.Strides[0] * batch + target.Strides[1] * 0;
        var strideBatchG = target.Strides[0] * batch + target.Strides[1] * 1;
        var strideBatchB = target.Strides[0] * batch + target.Strides[1] * 2;
        var strideY = target.Strides[2];
        var strideX = target.Strides[3];

        // Get a span of the whole tensor for fast access
        var tensorSpan = target.Buffer;

        // Try get continuous memory block of the entire image data
        if (image.DangerousTryGetSinglePixelMemory(out var memory))
        {
            Parallel.For(0, width * height, index =>
            {
                int x = index % width;
                int y = index / width;
                int tensorIndex = strideBatchR + strideY * (y + yPadding) + strideX * (x + xPadding);

                var pixel = memory.Span[index];
                WritePixel(tensorSpan.Span, tensorIndex, pixel, strideBatchR, strideBatchG, strideBatchB);
            });
        }
        else
        {
            Parallel.For(0, height, y =>
            {
                var rowSpan = image.DangerousGetPixelRowMemory(y).Span;
                int tensorYIndex = strideBatchR + strideY * (y + yPadding);

                for (int x = 0; x < width; x++)
                {
                    int tensorIndex = tensorYIndex + strideX * (x + xPadding);
                    var pixel = rowSpan[x];
                    WritePixel(tensorSpan.Span, tensorIndex, pixel, strideBatchR, strideBatchG, strideBatchB);
                }
            });
        }
    }

    private static void WritePixel(Span<float> tensorSpan, int tensorIndex, Rgb24 pixel, int strideBatchR, int strideBatchG, int strideBatchB)
    {
        tensorSpan[tensorIndex] = pixel.R / 255f;
        tensorSpan[tensorIndex + strideBatchG - strideBatchR] = pixel.G / 255f;
        tensorSpan[tensorIndex + strideBatchB - strideBatchR] = pixel.B / 255f;
    }
}