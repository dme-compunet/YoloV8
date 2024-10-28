namespace Compunet.YoloV8.Services;

internal class SegmentationParser(YoloMetadata metadata,
                                  IImageAdjustmentService imageAdjustment,
                                  IMemoryAllocatorService memoryAllocator,
                                  IRawBoundingBoxParser rawBoundingBoxParser) : IParser<Segmentation>
{
    public Segmentation[] ProcessTensorToResult(IYoloRawOutput output, Size size)
    {
        var adjustment = imageAdjustment.Calculate(size);

        var output0 = output.Output0;
        var output1 = output.Output1 ?? throw new Exception();

        var output0Span = output0.Buffer.Span;
        var output1Span = output1.Buffer.Span;

        var output0Strides = output0.Strides;
        var output1Strides = output1.Strides;

        var maskWidth = output1.Dimensions[3];
        var maskHeight = output1.Dimensions[2];
        var maskChannels = output1.Dimensions[1];

        var maskPaddingX = adjustment.Padding.X * maskWidth / metadata.ImageSize.Width;
        var maskPaddingY = adjustment.Padding.Y * maskHeight / metadata.ImageSize.Height;

        maskWidth -= maskPaddingX * 2;
        maskHeight -= maskPaddingY * 2;

        using var rawMaskBuffer = memoryAllocator.Allocate<float>(maskWidth * maskHeight);
        using var weightsBuffer = memoryAllocator.Allocate<float>(maskChannels);

        var weightsSpan = weightsBuffer.Memory.Span;
        var rawMaskBitmap = new BitmapBuffer(rawMaskBuffer.Memory, maskWidth, maskHeight);

        var boxes = rawBoundingBoxParser.Parse<RawBoundingBox>(output0);

        var result = new Segmentation[boxes.Length];

        for (var index = 0; index < boxes.Length; index++)
        {
            var box = boxes[index];
            var boxIndex = box.Index;

            var bounds = imageAdjustment.Adjust(box.Bounds, adjustment);

            // Collect the weights for this box
            for (var i = 0; i < maskChannels; i++)
            {
                var bufferIndex = GetIndex(output0Strides, 0, metadata.Names.Length + 4 + i, boxIndex);

                weightsSpan[i] = output0Span[bufferIndex];
            }

            rawMaskBitmap.Clear();

            for (var y = 0; y < rawMaskBitmap.Height; y++)
            {
                for (var x = 0; x < rawMaskBitmap.Width; x++)
                {
                    var value = 0f;

                    for (var i = 0; i < maskChannels; i++)
                    {
                        var bufferIndex = GetIndex(output1Strides, 0, i, y + maskPaddingY, x + maskPaddingX);

                        value += output1Span[bufferIndex] * weightsSpan[i];
                    }

                    rawMaskBitmap[y, x] = Sigmoid(value);
                }
            }

            var mask = new BitmapBuffer(bounds.Width, bounds.Height);

            ResizeAndCrop(rawMaskBitmap, mask, bounds, size);

            result[index] = new Segmentation
            {
                Mask = mask,
                Name = metadata.Names[box.NameIndex],
                Bounds = bounds,
                Confidence = box.Confidence,
            };
        }

        return result;
    }

    private static void ResizeAndCrop(BitmapBuffer source, BitmapBuffer target, Rectangle bounds, Size imageSize)
    {
        for (var y = 0; y < bounds.Height; y++)
        {
            for (var x = 0; x < bounds.Width; x++)
            {
                var sourceX = (float)(x + bounds.X) * (source.Width - 1) / (imageSize.Width - 1);
                var sourceY = (float)(y + bounds.Y) * (source.Height - 1) / (imageSize.Height - 1);

                var x0 = (int)sourceX;
                var y0 = (int)sourceY;

                var x1 = Math.Min(x0 + 1, source.Width - 1);
                var y1 = Math.Min(y0 + 1, source.Height - 1);

                var xLerp = sourceX - x0;
                var yLerp = sourceY - y0;

                var top = Lerp(source[y0, x0], source[y0, x1], xLerp);
                var bottom = Lerp(source[y1, x0], source[y1, x1], xLerp);

                target[y, x] = Lerp(top, bottom, yLerp);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetIndex(ReadOnlySpan<int> strides, int index0, int index1, int index2)
    {
        Debug.Assert(strides.Length == 3);

        return (strides[0] * index0) + (strides[1] * index1) + (strides[2] * index2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetIndex(ReadOnlySpan<int> strides, int index0, int index1, int index2, int index3)
    {
        Debug.Assert(strides.Length == 4);

        return (strides[0] * index0) + (strides[1] * index1) + (strides[2] * index2) + (strides[3] * index3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float value) => 1 / (1 + MathF.Exp(-value));
}