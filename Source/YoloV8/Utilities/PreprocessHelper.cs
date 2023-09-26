namespace Compunet.YoloV8.Utilities;

internal static class PreprocessHelper
{
    public static void ProcessToTensor(Image<Rgb24> image, Size modelSize, bool originalAspectRatio, Tensor<float> target, int batch)
    {
        int xPadding;
        int yPadding;

        #region Resize

        int targetWidth;
        int targetHeight;

        if (originalAspectRatio)
        {
            var xRatio = (float)modelSize.Width / image.Width;
            var yRatio = (float)modelSize.Height / image.Height;

            var ratio = Math.Min(xRatio, yRatio);

            targetWidth = (int)(image.Width * ratio);
            targetHeight = (int)(image.Height * ratio);

            xPadding = (modelSize.Width - targetWidth) / 2;
            yPadding = (modelSize.Height - targetHeight) / 2;
        }
        else
        {
            targetWidth = modelSize.Width;
            targetHeight = modelSize.Height;

            xPadding = 0;
            yPadding = 0;
        }

        image.Mutate(x => x.Resize(targetWidth, targetHeight));

        #endregion

        #region Copy To Tensor

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

        #endregion
    }
}