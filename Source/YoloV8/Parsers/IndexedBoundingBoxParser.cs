namespace Compunet.YoloV8.Parsers;

internal readonly ref struct IndexedBoundingBoxParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
{
    private readonly ArrayPool<IndexedBoundingBox> _boxesArrayPool = ArrayPool<IndexedBoundingBox>.Create();

    public IndexedBoundingBox[] Parse(Tensor<float> output, Size originSize)
    {
        int xPadding;
        int yPadding;

        if (configuration.KeepOriginalAspectRatio)
        {
            var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)originSize.Width,
                                          metadata.ImageSize.Height / (float)originSize.Height);

            xPadding = (int)((metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2);
            yPadding = (int)((metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2);
        }
        else
        {
            xPadding = 0;
            yPadding = 0;
        }

        return Parse(output, originSize, xPadding, yPadding);
    }

    public IndexedBoundingBox[] Parse(Tensor<float> output, Size originSize, int xPadding, int yPadding)
    {
        var xRatio = (float)originSize.Width / metadata.ImageSize.Width;
        var yRatio = (float)originSize.Height / metadata.ImageSize.Height;

        if (configuration.KeepOriginalAspectRatio)
        {
            var maxRatio = Math.Max(xRatio, yRatio);

            xRatio = maxRatio;
            yRatio = maxRatio;
        }

        return Parse(output, originSize, xPadding, yPadding, xRatio, yRatio);
    }

    public IndexedBoundingBox[] Parse(Tensor<float> output, Size originSize, int xPadding, int yPadding, float xRatio, float yRatio)
    {
        var _metadata = metadata;
        var _configuration = configuration;

        var boxesCount = output.Dimensions[2];
        var boxes = _boxesArrayPool.Rent(boxesCount);

        try
        {
            Parallel.For(0, boxesCount, i =>
            {
                for (int j = 0; j < _metadata.Names.Count; j++)
                {
                    var confidence = output[0, j + 4, i];

                    if (confidence <= _configuration.Confidence)
                    {
                        continue;
                    }

                    var x = output[0, 0, i];
                    var y = output[0, 1, i];
                    var w = output[0, 2, i];
                    var h = output[0, 3, i];

                    var xMin = (int)((x - w / 2 - xPadding) * xRatio);
                    var yMin = (int)((y - h / 2 - yPadding) * yRatio);
                    var xMax = (int)((x + w / 2 - xPadding) * xRatio);
                    var yMax = (int)((y + h / 2 - yPadding) * yRatio);

                    xMin = Math.Clamp(xMin, 0, originSize.Width);
                    yMin = Math.Clamp(yMin, 0, originSize.Height);
                    xMax = Math.Clamp(xMax, 0, originSize.Width);
                    yMax = Math.Clamp(yMax, 0, originSize.Height);

                    var name = _metadata.Names[j];
                    var bounds = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);

                    if (bounds.Width == 0 || bounds.Height == 0)
                    {
                        continue;
                    }

                    boxes[i] = new IndexedBoundingBox
                    {
                        Index = i,
                        Class = name,
                        Bounds = bounds,
                        Confidence = confidence
                    };
                }
            });

            return NonMaxSuppressionHelper.Suppress(GetActiveBoxes(boxes, boxesCount), configuration.IoU);
        }
        finally
        {
            _boxesArrayPool.Return(boxes, true);
        }
    }

    private static IndexedBoundingBox[] GetActiveBoxes(IndexedBoundingBox[] boxes, int boxesCount)
    {
        var activeCount = 0;

        for (var i = 0; i < boxesCount; i++)
        {
            if (boxes[i].IsEmpty == false)
            {
                activeCount++;
            }
        }

        var activeIndex = 0;
        var activeBoxes = new IndexedBoundingBox[activeCount];

        for (var i = 0; i < boxesCount; i++)
        {
            var box = boxes[i];

            if (box.IsEmpty)
            {
                continue;
            }

            activeBoxes[activeIndex++] = box;
        }

        return activeBoxes;
    }
}