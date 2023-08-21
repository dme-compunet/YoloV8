using Compunet.YoloV8.Data;
using Compunet.YoloV8.Extensions;
using Compunet.YoloV8.Metadata;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Compunet.YoloV8.Parsers;

internal readonly struct SegmentationOutputParser
{
    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    private record struct IndexedBoundingBox(int Index, YoloV8Class Name, Rectangle Bounds, float Confidence);

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

        var maskChannelCount = output0.Dimensions[1] - 4 - metadata.Classes.Count;

        var boxes = new List<IndexedBoundingBox>(output0.Dimensions[2]);

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

                var bounds = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);
                var name = metadata.Classes[j];

                var box = new IndexedBoundingBox(i, name, bounds, confidence);
                boxes.Add(box);
            }
        });

        var selected = boxes.NonMaxSuppression(x => x.Bounds,
                                               x => x.Confidence,
                                               _parameters.IoU);

        var result = new SegmentationBoundingBox[selected.Count];

        Parallel.For(0, selected.Count, index =>
        {
            var box = selected[index];

            var maskWeights = GetMaskWeights(output0, box.Index, maskChannelCount, metadata.Classes.Count + 4);

            var mask = ProcessMask(output1, maskWeights, box.Bounds, originSize, metadata.ImageSize, xPadding, yPadding);

            var value = new SegmentationBoundingBox(box.Name, box.Bounds, box.Confidence, mask);

            result[index] = value;
        });

        return result;
    }

    private static IMask ProcessMask(Tensor<float> maskPrototypes, ReadOnlySpan<float> maskWeights, Rectangle bounds, Size originSize, Size modelSize, int xPadding, int yPadding)
    {
        var maskChannels = maskPrototypes.Dimensions[1];
        var maskHeight = maskPrototypes.Dimensions[2];
        var maskWidth = maskPrototypes.Dimensions[3];

        if (maskChannels != maskWeights.Length)
            throw new InvalidOperationException();

        using var bitmap = new Image<L8>(maskWidth, maskHeight);

        for (int y = 0; y < maskHeight; y++)
        {
            for (int x = 0; x < maskWidth; x++)
            {
                var value = 0F;

                for (int i = 0; i < maskChannels; i++)
                    value += maskPrototypes[0, i, y, x] * maskWeights[i];

                value = Sigmoid(value);

                var color = GetLuminance(value);
                var pixel = new L8(color);

                bitmap[x, y] = pixel;
            }
        }

        var xPad = xPadding * maskWidth / modelSize.Width;
        var yPad = yPadding * maskHeight / modelSize.Height;

        var paddingCropRectangle = new Rectangle(xPad,
                                                 yPad,
                                                 maskWidth - xPad * 2,
                                                 maskHeight - yPad * 2);

        bitmap.Mutate(x =>
        {
            // crop for preprocess resize padding
            x.Crop(paddingCropRectangle);

            // resize to original image size
            x.Resize(originSize);

            // crop for getting the object segmentation only
            x.Crop(bounds);
        });

        var final = new float[bounds.Width, bounds.Height];

        bitmap.ForEachPixel((point, pixel) =>
        {
            var confidence = GetConfidence(pixel.PackedValue);
            final[point.X, point.Y] = confidence;
        });

        return new Mask(final);
    }

    private static ReadOnlySpan<float> GetMaskWeights(Tensor<float> output, int boxIndex, int maskChannelCount, int maskWeightsOffset) 
    {
        var maskWeights = new float[maskChannelCount];

        for (int i = 0; i < maskChannelCount; i++)
        {
            maskWeights[i] = output[0, maskWeightsOffset + i, boxIndex];
        }

        return maskWeights;
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