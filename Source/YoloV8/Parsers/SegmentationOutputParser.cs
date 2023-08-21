using Compunet.YoloV8.Data;
using Compunet.YoloV8.Extensions;
using Compunet.YoloV8.Metadata;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Compunet.YoloV8.Parsers;

internal readonly struct SegmentationOutputParser
{
    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    public SegmentationOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
    {
        _metadata = metadata;
        _parameters = parameters;
    }

    public IReadOnlyList<ISegmentationBoundingBox> Parse(IReadOnlyList<Tensor<float>> outputs, Size originSize)
    {
        var metadata = _metadata;
        var parameters = _parameters;

        var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)originSize.Width, metadata.ImageSize.Height / (float)originSize.Height);

        var xPadding = (int)((metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2);
        var yPadding = (int)((metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2);

        var magnificationRatio = Math.Max((float)originSize.Width / metadata.ImageSize.Width, (float)originSize.Height / metadata.ImageSize.Height);

        var output0 = outputs[0];
        var output1 = outputs[1];

        var boxes = new List<SegmentationBoundingBox>(output0.Dimensions[2]);

        Parallel.For(0, output0.Dimensions[2], i =>
        {
            for (int j = 0; j < metadata.Classes.Count; j++)
            {
                var confidence = output0[0, j + 4, i];

                if (confidence <= parameters.Confidence)
                    continue;

                var x = output0[0, 0, i];
                var y = output0[0, 1, i];
                var w = output0[0, 2, i];
                var h = output0[0, 3, i];

                var xMin = (int)((x - w / 2 - xPadding) * magnificationRatio);
                var yMin = (int)((y - h / 2 - yPadding) * magnificationRatio);
                var xMax = (int)((x + w / 2 - xPadding) * magnificationRatio);
                var yMax = (int)((y + h / 2 - yPadding) * magnificationRatio);

                xMin = Math.Clamp(xMin, 0, originSize.Width);
                yMin = Math.Clamp(yMin, 0, originSize.Height);
                xMax = Math.Clamp(xMax, 0, originSize.Width);
                yMax = Math.Clamp(yMax, 0, originSize.Height);

                var rectangle = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);
                var name = metadata.Classes[j];

                var maskChannelCount = output0.Dimensions[1] - 4 - metadata.Classes.Count;

                var maskWeights = new float[maskChannelCount];

                for (int k = 0; k < maskChannelCount; k++)
                {
                    var offset = 4 + metadata.Classes.Count + k;
                    maskWeights[k] = output0[0, offset, i];
                }

                var mask = ProcessMask(output1, maskWeights, rectangle, originSize, metadata.ImageSize, xPadding, yPadding);

                var box = new SegmentationBoundingBox(name, rectangle, confidence, mask);
                boxes.Add(box);
            }
        });

        var selected = boxes.NonMaxSuppression(x => x.Rectangle,
                                               x => x.Confidence,
                                               _parameters.IoU);

        return selected;
    }

    private static IMask ProcessMask(Tensor<float> prototypes, float[] weights, Rectangle rectangle, Size originSize, Size modelSize, int xPadding, int yPadding)
    {
        var maskChannels = prototypes.Dimensions[1];
        var maskHeight = prototypes.Dimensions[2];
        var maskWidth = prototypes.Dimensions[3];

        if (maskChannels != weights.Length)
            throw new InvalidOperationException();

        using var bitmap = new Image<L8>(maskWidth, maskHeight);

        for (int x = 0; x < maskWidth; x++)
        {
            for (int y = 0; y < maskHeight; y++)
            {
                var value = 0F;

                for (int i = 0; i < maskChannels; i++)
                    value += prototypes[0, i, x, y] * weights[i];

                value = Sigmoid(value);

                var color = GetLuminance(value);
                var pixel = new L8(color);

                bitmap[x, y] = pixel;
            }
        }

        bitmap.Mutate(x =>
        {
            x.RotateFlip(RotateMode.Rotate90, FlipMode.Horizontal);

            var xPad = xPadding * maskWidth / modelSize.Width;
            var yPad = yPadding * maskHeight / modelSize.Height;

            var crop = new Rectangle(xPad,
                                     yPad,
                                     maskWidth - xPad * 2,
                                     maskHeight - yPad * 2);
            x.Crop(crop);

            x.Resize(originSize);
            x.Crop(rectangle);
        });

        var final = new float[rectangle.Width, rectangle.Height];

        bitmap.ForEachPixel((point, pixel) =>
        {
            var confidence = GetConfidence(pixel.PackedValue);
            final[point.X, point.Y] = confidence;
        });

        return new Mask(final);
    }

    #region Helpers

    private static float Sigmoid(float value)
    {
        var k = MathF.Exp(value);
        return k / (1.0f + k);
    }

    private static byte GetLuminance(float confidence)
    {
        return (byte)((confidence * 255 - 255) * -1);
    }

    private static float GetConfidence(byte luminance)
    {
        return (luminance - 255) * -1 / 255F;
    }

    #endregion
}