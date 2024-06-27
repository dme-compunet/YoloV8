namespace Compunet.YoloV8.Parsers;

internal readonly struct SegmentationOutputParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
{
    public SegmentationBoundingBox[] Parse(Tensor<float> boxesOutput, Tensor<float> maskPrototypes, Size originSize)
    {
        var _metadata = metadata;

        int xPadding;
        int yPadding;

        if (configuration.KeepOriginalAspectRatio)
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

        var maskChannelCount = boxesOutput.Dimensions[1] - 4 - metadata.Names.Count;

        var boxes = new IndexedBoundingBoxParser(_metadata, configuration).Parse(boxesOutput, originSize, xPadding, yPadding);

        var result = new SegmentationBoundingBox[boxes.Length];

        for (int index = 0; index < boxes.Length; index++)
        {
            var box = boxes[index];

            var maskWeights = ExtractMaskWeights(boxesOutput, box.Index, maskChannelCount, _metadata.Names.Count + 4);

            var mask = ProcessMask(maskPrototypes, maskWeights, box.Bounds, originSize, _metadata.ImageSize, xPadding, yPadding);

            result[index] = new SegmentationBoundingBox
            {
                Mask = mask,
                Class = box.Class,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };
        }

        return result;
    }

    private static SegmentationMask ProcessMask(Tensor<float> prototypes,
                                                float[] weights,
                                                Rectangle bounds,
                                                Size originSize,
                                                Size modelSize,
                                                int xPadding,
                                                int yPadding)
    {
        var maskChannels = prototypes.Dimensions[1];
        var maskHeight = prototypes.Dimensions[2];
        var maskWidth = prototypes.Dimensions[3];

        if (maskChannels != weights.Length)
        {
            throw new InvalidOperationException();
        }

        using var bitmap = new Image<L8>(maskWidth, maskHeight);

        for (int y = 0; y < maskHeight; y++)
        {
            for (int x = 0; x < maskWidth; x++)
            {
                var value = 0F;

                for (int i = 0; i < maskChannels; i++)
                    value += prototypes[0, i, y, x] * weights[i];

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

        bitmap.EnumeratePixels((point, pixel) =>
        {
            var confidence = GetConfidence(pixel.PackedValue);
            final[point.X, point.Y] = confidence;
        });

        return new SegmentationMask
        {
            Mask = final
        };
    }

    private static float[] ExtractMaskWeights(Tensor<float> output, int boxIndex, int maskChannelCount, int maskWeightsOffset)
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
        //return 1 / (1 + MathF.Exp(-value));

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