namespace Compunet.YoloSharp.Parsing.Parsers;

internal class RawBoundingBoxParser(YoloMetadata metadata,
                                    YoloConfiguration configuration,
                                    IMemoryAllocatorService memoryAllocator,
                                    INonMaxSuppressionService nonMaxSuppression) : IRawBoundingBoxParser
{
    private T[] ParseYoloV8<T>(MemoryTensor<float> tensor) where T : IRawBoundingBox<T>
    {
        var stride1 = tensor.Strides[1];
        var boxesCount = tensor.Dimensions[2];
        var namesCount = metadata.Names.Length;

        var boxes = memoryAllocator.Allocate<T>(boxesCount);
        var boxesIndex = 0;
        var boxesSpan = boxes.Memory.Span;
        var tensorSpan = tensor.Buffer.Span;

        var context = new RawParsingContext
        {
            Tensor = tensor,
            NameCount = namesCount,
            Architecture = YoloArchitecture.YoloV8Or11,
        };

        for (var boxIndex = 0; boxIndex < boxesCount; boxIndex++)
        {
            for (var nameIndex = 0; nameIndex < namesCount; nameIndex++)
            {
                var confidence = tensorSpan[(nameIndex + 4) * stride1 + boxIndex];

                if (confidence <= configuration.Confidence)
                {
                    continue;
                }

                var box = T.Parse(ref context, boxIndex, nameIndex, confidence);

                if (box.Bounds.Width == 0 || box.Bounds.Height == 0)
                {
                    continue;
                }

                boxesSpan[boxesIndex++] = box;
            }
        }

        return nonMaxSuppression.Suppress(boxesSpan[..boxesIndex], configuration.IoU);
    }

    private T[] ParseYoloV10<T>(MemoryTensor<float> tensor) where T : IRawBoundingBox<T>
    {
        var stride1 = tensor.Strides[1];
        var stride2 = tensor.Strides[2];

        var boxesCount = tensor.Dimensions[1];
        var boxes = memoryAllocator.Allocate<T>(boxesCount);
        var boxesIndex = 0;
        var boxesSpan = boxes.Memory.Span;
        var tensorSpan = tensor.Buffer.Span;

        var context = new RawParsingContext
        {
            Tensor = tensor,
            Architecture = YoloArchitecture.YoloV10,
        };

        for (var index = 0; index < boxesCount; index++)
        {
            var boxOffset = index * stride1;

            var confidence = tensorSpan[boxOffset + 4 * stride2];

            if (confidence <= configuration.Confidence)
            {
                continue;
            }

            var nameIndex = (int)tensorSpan[boxOffset + 5 * stride2];
            var box = T.Parse(ref context, index, nameIndex, confidence);

            if (box.Bounds.Width == 0 || box.Bounds.Height == 0)
            {
                continue;
            }

            boxesSpan[boxesIndex++] = box;
        }

        return nonMaxSuppression.Suppress(boxesSpan[..boxesIndex], configuration.IoU);
    }

    public T[] Parse<T>(MemoryTensor<float> tensor) where T : IRawBoundingBox<T>
    {
        if (metadata.Architecture == YoloArchitecture.YoloV10)
        {
            return ParseYoloV10<T>(tensor);
        }

        return ParseYoloV8<T>(tensor);
    }
}