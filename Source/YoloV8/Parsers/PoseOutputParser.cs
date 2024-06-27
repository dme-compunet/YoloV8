namespace Compunet.YoloV8.Parsers;

internal readonly ref struct PoseOutputParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
{
    public PoseBoundingBox[] Parse(Tensor<float> output, Size originSize)
    {
        var poseMetadata = (YoloV8PoseMetadata)metadata;

        int xPadding;
        int yPadding;

        var xRatio = (float)originSize.Width / metadata.ImageSize.Width;
        var yRatio = (float)originSize.Height / metadata.ImageSize.Height;

        if (configuration.KeepOriginalAspectRatio)
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

        var boxes = new IndexedBoundingBoxParser(metadata, configuration).Parse(output, originSize, xPadding, yPadding, xRatio, yRatio);

        var shape = poseMetadata.KeypointShape;

        var result = new PoseBoundingBox[boxes.Length];

        for (int index = 0; index < boxes.Length; index++)
        {
            var box = boxes[index];

            var keypoints = new Keypoint[shape.Count];

            for (int i = 0; i < shape.Count; i++)
            {
                var offset = i * shape.Channels + 4 + poseMetadata.Names.Count;

                var pointX = (int)((output[0, offset + 0, box.Index] - xPadding) * xRatio);
                var pointY = (int)((output[0, offset + 1, box.Index] - yPadding) * yRatio);

                var pointConfidence = poseMetadata.KeypointShape.Channels switch
                {
                    2 => 1F,
                    3 => output[0, offset + 2, box.Index],
                    _ => throw new NotSupportedException("Unexpected keypoint shape")
                };

                keypoints[i] = new Keypoint
                {
                    Index = i,
                    Point = new Point(pointX, pointY),
                    Confidence = pointConfidence
                };
            }

            result[index] = new PoseBoundingBox
            {
                Class = box.Class,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
                Keypoints = keypoints
            };
        }

        return result;
    }
}