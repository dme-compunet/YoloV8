namespace Compunet.YoloV8.Services;

internal class SegmentationParser(YoloMetadata metadata,
                                  YoloConfiguration configuration,
                                  IMemoryAllocatorService memoryAllocator,
                                  IRawBoundingBoxParser rawBoundingBoxParser) : IParser<Segmentation>
{
    public Segmentation[] ProcessTensorToResult(YoloRawOutput output, Size size)
    {
        var xPadding = 0;
        var yPadding = 0;

        if (configuration.KeepAspectRatio)
        {
            var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)size.Width,
                                          metadata.ImageSize.Height / (float)size.Height);

            xPadding = (int)((metadata.ImageSize.Width - size.Width * reductionRatio) / 2);
            yPadding = (int)((metadata.ImageSize.Height - size.Height * reductionRatio) / 2);
        }

        var output0 = output.Output0;
        var output1 = output.Output1 ?? throw new Exception();

        var boxes = rawBoundingBoxParser.Parse<RawBoundingBox>(output0, size);
        var maskChannelCount = output0.Dimensions[1] - 4 - metadata.Names.Length;

        var result = new Segmentation[boxes.Length];

        for (var index = 0; index < boxes.Length; index++)
        {
            var box = boxes[index];

            using var maskWeights = CollectMaskWeights(output0, box.Index, maskChannelCount, metadata.Names.Length + 4);

            var mask = ProcessMask(output1, maskWeights.Memory.Span, box.Bounds, size, metadata.ImageSize, xPadding, yPadding);

            result[index] = new Segmentation
            {
                Mask = mask,
                Name = box.Name,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };
        }

        return result;
    }

    private static SegmentationMask ProcessMask(Tensor<float> prototypes,
                                                ReadOnlySpan<float> weights,
                                                Rectangle bounds,
                                                Size imageSize,
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

        for (var y = 0; y < maskHeight; y++)
        {
            for (var x = 0; x < maskWidth; x++)
            {
                var value = 0F;

                for (int i = 0; i < maskChannels; i++)
                {
                    value += prototypes[0, i, y, x] * weights[i];
                }

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
            x.Resize(imageSize);

            // Crop for getting the object segmentation only
            x.Crop(bounds);
        });

        return CreateMaskFromBitmap(bitmap);

        //var mask = new float[bounds.Width, bounds.Height];

        //bitmap.EnumeratePixels((point, pixel) =>
        //{
        //    mask[point.X, point.Y] = GetConfidence(pixel.PackedValue);
        //});

        //return new SegmentationMask
        //{
        //    Mask = mask
        //};
    }


    private IMemoryOwner<float> CollectMaskWeights(Tensor<float> output, int boxIndex, int maskChannelCount, int maskWeightsOffset)
    {
        var weights = memoryAllocator.Allocate<float>(maskChannelCount);
        var weightsSpan = weights.Memory.Span;

        for (int i = 0; i < maskChannelCount; i++)
        {
            weightsSpan[i] = output[0, maskWeightsOffset + i, boxIndex];
        }

        return weights;
    }

    private static SegmentationMask CreateMaskFromBitmap(Image<L8> bitmap)
    {
        var mask = new float[bitmap.Width, bitmap.Height];

        bitmap.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < bitmap.Height; y++)
            {
                var row = accessor.GetRowSpan(y);

                for (var x = 0; x < bitmap.Width; x++)
                {
                    mask[x, y] = GetConfidence(row[x].PackedValue);
                }
            }
        });

        return new SegmentationMask
        {
            Mask = mask
        };
    }

    #region Helpers

    private static float Sigmoid(float value) => 1 / (1 + MathF.Exp(-value));

    private static byte GetLuminance(float confidence) => (byte)((confidence * 255 - 255) * -1);

    private static float GetConfidence(byte luminance) => (luminance - 255) * -1 / 255F;

    #endregion
}