namespace Compunet.YoloV8.Parsers;

internal readonly struct SegmentationOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
{
    private readonly YoloV8Metadata _metadata = metadata;
    private readonly YoloV8Parameters _parameters = parameters;

    public IEnumerable<SegmentationBoundingBox> Parse(IEnumerable<Tensor<float>> outputs, Size originSize)
    {
        var metadata = _metadata;

        int xPadding;
        int yPadding;

        if (_parameters.KeepOriginalAspectRatio)
        {
            var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)originSize.Width, metadata.ImageSize.Height / (float)originSize.Height);

            xPadding = (int)((metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2);
            yPadding = (int)((metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2);
        }
        else
        {
            xPadding = 0;
            yPadding = 0;
        }

        var output0 = outputs.ElementAt(0);
        var output1 = outputs.ElementAt(1);

        var maskChannelCount = output0.Dimensions[1] - 4 - metadata.Classes.Count;

        var boxes = new IndexedBoundingBoxParser(_metadata, _parameters).Parse(output0, originSize, xPadding, yPadding);

        return boxes.AsParallel().Select(box =>
        {
            var maskWeights = ExtractMaskWeights(output0, box.Index, maskChannelCount, metadata.Classes.Count + 4);

            var mask = ProcessMask(output1, maskWeights, box.Bounds, originSize, metadata.ImageSize, xPadding, yPadding);

            var value = new SegmentationBoundingBox
            {
                Mask = mask,
                Class = box.Class,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };

            return value;
        });
    }

    private static SegmentationMask ProcessMask(Tensor<float> maskPrototypes,
                                                ReadOnlySpan<float> maskWeights,
                                                Rectangle bounds,
                                                Size originSize,
                                                Size modelSize,
                                                int xPadding,
                                                int yPadding)
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
            // Crop for preprocess resize padding
            x.Crop(paddingCropRectangle);

            // Resize to original image size
            x.Resize(originSize);

            // Crop for getting the object segmentation only
            x.Crop(bounds);
        });

        var final = new float[bounds.Width, bounds.Height];

        bitmap.IteratePixels((point, pixel) =>
        {
            var confidence = GetConfidence(pixel.PackedValue);
            final[point.X, point.Y] = confidence;
        });

        return new SegmentationMask
        {
            Mask = final
        };
    }

    private static ReadOnlySpan<float> ExtractMaskWeights(Tensor<float> output, int boxIndex, int maskChannelCount, int maskWeightsOffset)
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