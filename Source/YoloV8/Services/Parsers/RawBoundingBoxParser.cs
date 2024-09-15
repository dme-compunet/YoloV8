namespace Compunet.YoloV8.Services;

internal class RawBoundingBoxParser(YoloMetadata metadata,
                                    YoloConfiguration configuration,
                                    IMemoryAllocatorService memoryAllocator,
                                    INonMaxSuppressionService nonMaxSuppression) : IRawBoundingBoxParser
{
    public T[] Parse<T>(DenseTensor<float> tensor, Size size) where T : IRawBoundingBox<T>
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

        return Parse<T>(tensor, size, new Vector<int>(xPadding, yPadding));
    }

    public T[] Parse<T>(DenseTensor<float> tensor, Size imageSize, Vector<int> padding) where T : IRawBoundingBox<T>
    {
        var xRatio = (float)imageSize.Width / metadata.ImageSize.Width;
        var yRatio = (float)imageSize.Height / metadata.ImageSize.Height;

        if (configuration.KeepAspectRatio)
        {
            var maxRatio = Math.Max(xRatio, yRatio);

            xRatio = maxRatio;
            yRatio = maxRatio;
        }

        return Parse<T>(tensor, padding, new Vector<float>(xRatio, yRatio));
    }

    public T[] Parse<T>(DenseTensor<float> tensor, Vector<int> padding, Vector<float> ratio) where T : IRawBoundingBox<T>
    {
        if (metadata.Architecture == YoloArchitecture.YoloV10)
        {
            return ParseYoloV10<T>(tensor, padding, ratio);
        }

        return ParseYoloV8<T>(tensor, padding, ratio);
    }

    private T[] ParseYoloV8<T>(DenseTensor<float> tensor, Vector<int> padding, Vector<float> ratio) where T : IRawBoundingBox<T>
    {
        var stride1 = tensor.Strides[1];
        var boxesCount = tensor.Dimensions[2];
        var namesCount = metadata.Names.Length;

        var boxes = memoryAllocator.Allocate<T>(boxesCount);

        var boxesSpan = boxes.Memory.Span;
        var tensorSpan = tensor.Buffer.Span;

        var context = new RawParsingContext
        {
            Tensor = tensor,
            Padding = padding,
            Ratio = ratio,
            Stride1 = stride1,
            NameCount = namesCount,
        };

        var nonEmptyBoxesCount = 0;

        for (var boxIndex = 0; boxIndex < boxesCount; boxIndex++)
        {
            var emptyBox = true;

            for (var nameIndex = 0; nameIndex < namesCount; nameIndex++)
            {
                var confidence = tensorSpan[(nameIndex + 4) * stride1 + boxIndex];

                if (confidence <= configuration.Confidence)
                {
                    continue;
                }

                emptyBox = false;

                var name = metadata.Names[nameIndex];
                var box = T.Parse(ref context, boxIndex, name, confidence, YoloArchitecture.YoloV8);

                if (box.Bounds.Width == 0 || box.Bounds.Height == 0)
                {
                    continue;
                }

                boxesSpan[boxIndex] = box;
            }

            if (emptyBox)
            {
                boxesSpan[boxIndex] = T.Empty;
            }
            else
            {
                nonEmptyBoxesCount++;
            }
        }

        using var nonEmptyBoxes = GetNonEmptyBoxes<T>(boxes.Memory.Span, nonEmptyBoxesCount);

        return nonMaxSuppression.Suppress(nonEmptyBoxes.Memory.Span, configuration.IoU);
    }

    private T[] ParseYoloV10<T>(DenseTensor<float> tensor, Vector<int> padding, Vector<float> ratio) where T : IRawBoundingBox<T>
    {
        var stride1 = tensor.Strides[1];
        var stride2 = tensor.Strides[2];

        var boxesCount = tensor.Dimensions[1];
        var boxes = memoryAllocator.Allocate<T>(boxesCount);

        var boxesSpan = boxes.Memory.Span;
        var tensorSpan = tensor.Buffer.Span;

        var context = new RawParsingContext
        {
            Tensor = tensor,
            Padding = padding,
            Ratio = ratio,
            Stride1 = stride1
        };

        var nonEmptyBoxesCount = 0;

        for (var index = 0; index < boxesCount; index++)
        {
            var boxOffset = index * stride1;

            var confidence = tensorSpan[boxOffset + 4 * stride2];

            if (confidence <= configuration.Confidence)
            {
                boxesSpan[index] = T.Empty;
                continue;
            }

            nonEmptyBoxesCount++;

            var name = metadata.Names[(int)tensorSpan[boxOffset + 5 * stride2]];
            var box = T.Parse(ref context, index, name, confidence, YoloArchitecture.YoloV10);

            if (box.Bounds.Width == 0 || box.Bounds.Height == 0)
            {
                continue;
            }

            boxesSpan[index] = box;
        }

        using var nonEmptyBoxes = GetNonEmptyBoxes<T>(boxes.Memory.Span, nonEmptyBoxesCount);

        return nonMaxSuppression.Suppress(nonEmptyBoxes.Memory.Span, configuration.IoU);
    }

    private IMemoryOwner<T> GetNonEmptyBoxes<T>(ReadOnlySpan<T> boxes, int count) where T : IRawBoundingBox<T>
    {
        //var activeCount = 0;

        //// Count the non-empty boxes
        //for (var i = 0; i < boxes.Length; i++)
        //{
        //    if (boxes[i].IsEmpty == false)
        //    {
        //        activeCount++;
        //    }
        //}

        var activeIndex = 0;
        var activeBoxes = memoryAllocator.Allocate<T>(count);

        for (var i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];

            if (box.IsEmpty)
            {
                continue;
            }

            activeBoxes.Memory.Span[activeIndex++] = box;
        }

        return activeBoxes;
    }
}