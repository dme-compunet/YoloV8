namespace Compunet.YoloV8.Parsers;

internal readonly struct PoseOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
{
    private readonly YoloV8Metadata _metadata = metadata;
    private readonly YoloV8Parameters _parameters = parameters;

    public IEnumerable<PoseBoundingBox> Parse(Tensor<float> output, Size originSize)
    {
        var metadata = (YoloV8PoseMetadata)_metadata;

        int xPadding;
        int yPadding;

        var xRatio = (float)originSize.Width / metadata.ImageSize.Width;
        var yRatio = (float)originSize.Height / metadata.ImageSize.Height;

        if (_parameters.KeepOriginalAspectRatio)
        {
            var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)originSize.Width, metadata.ImageSize.Height / (float)originSize.Height);

            xPadding = (int)((metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2);
            yPadding = (int)((metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2);

            var maxRatio = Math.Max(xRatio, yRatio);

            xRatio = maxRatio;
            yRatio = maxRatio;
        }
        else
        {
            xPadding = 0;
            yPadding = 0;
        }

        var boxes = new IndexedBoundingBoxParser(_metadata, _parameters).Parse(output, originSize, xPadding, yPadding, xRatio, yRatio);

        var shape = metadata.KeypointShape;

        return boxes.AsParallel().Select(box =>
        {
            var keypoints = new Keypoint[shape.Count];

            for (int index = 0; index < shape.Count; index++)
            {
                var offset = index * shape.Channels + 4 + metadata.Classes.Count;

                var pointX = (int)((output[0, offset + 0, box.Index] - xPadding) * xRatio);
                var pointY = (int)((output[0, offset + 1, box.Index] - yPadding) * yRatio);

                var pointConfidence = metadata.KeypointShape.Channels switch
                {
                    2 => 1F,
                    3 => output[0, offset + 2, box.Index],
                    _ => throw new NotSupportedException("Unexpected keypoint shape")
                };

                keypoints[index] = new Keypoint
                {
                    Index = index,
                    Point = new Point(pointX, pointY),
                    Confidence = pointConfidence
                };
            }

            return new PoseBoundingBox
            {
                Class = box.Class,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
                Keypoints = keypoints
            };
        });
    }
}