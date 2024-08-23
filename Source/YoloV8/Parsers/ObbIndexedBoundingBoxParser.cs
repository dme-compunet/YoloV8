// https://github.com/ultralytics/ultralytics/issues/7667

namespace Compunet.YoloV8;

internal readonly ref struct ObbIndexedBoundingBoxParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
{
    private static readonly ArrayPool<ObbIndexedBoundingBox> _boxesArrayPool = ArrayPool<ObbIndexedBoundingBox>.Create();

    public ObbIndexedBoundingBox[] Parse(Tensor<float> output, Size originSize)
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

    public ObbIndexedBoundingBox[] Parse(Tensor<float> output, Size originSize, int xPadding, int yPadding)
    {
        var xRatio = (float)originSize.Width / metadata.ImageSize.Width;
        var yRatio = (float)originSize.Height / metadata.ImageSize.Height;

        if (configuration.KeepOriginalAspectRatio)
        {
            var maxRatio = Math.Max(xRatio, yRatio);

            xRatio = maxRatio;
            yRatio = maxRatio;
        }

        return Parse(output, xPadding, yPadding, xRatio, yRatio);
    }

    public ObbIndexedBoundingBox[] Parse(Tensor<float> output, int xPadding, int yPadding, float xRatio, float yRatio)
    {
        var _metadata = metadata;
        var _parameters = configuration;

        var detectionDataSize = output.Dimensions[1];
        var boxesCount = output.Dimensions[2];
        var boxes = _boxesArrayPool.Rent(boxesCount);

        try
        {
            Parallel.For(0, boxesCount, i =>
            {
                var maxConfidence = _parameters.Confidence;
                var maxConfidenceIndex = -1;

                for (int j = 0; j < _metadata.Names.Count; j++)
                {
                    var confidence = output[0, j + 4, i];

                    if (confidence > maxConfidence)
                    {
                        maxConfidence = confidence;
                        maxConfidenceIndex = j;
                    }
                }

                if (maxConfidenceIndex == -1)
                {
                    return;
                }

                var x = (int)((output[0, 0, i] - xPadding) * xRatio);
                var y = (int)((output[0, 1, i] - yPadding) * yRatio);
                var w = (int)(output[0, 2, i] * xRatio);
                var h = (int)(output[0, 3, i] * yRatio);

                var bounds = new Rectangle(x, y, w, h);

                var angle = (output[0, detectionDataSize - 1, i]); // Radians

                // Angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
                if (angle >= MathF.PI && angle <= 0.75 * MathF.PI)
                {
                    angle -= MathF.PI;
                }

                var name = _metadata.Names[maxConfidenceIndex];

                boxes[i] = new ObbIndexedBoundingBox
                {
                    Index = i,
                    Class = name,
                    Bounds = bounds,
                    Angle = angle * 180 / MathF.PI, // Degrees
                    Confidence = maxConfidence
                };
            });

            return ObbNonMaxSuppressionHelper.Suppress(GetActiveBoxes(boxes, boxesCount), configuration.IoU);
        }
        finally
        {
            _boxesArrayPool.Return(boxes, true);
        }
    }
    
    private static ObbIndexedBoundingBox[] GetActiveBoxes(ObbIndexedBoundingBox[] boxes, int boxesCount)
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
        var activeBoxes = new ObbIndexedBoundingBox[activeCount];

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