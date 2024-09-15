namespace Compunet.YoloV8.Services;

internal class PoseParser(YoloPoseMetadata metadata,
                          YoloConfiguration configuration,
                          IRawBoundingBoxParser rawBoundingBoxParser) : IParser<Pose>
{
    public Pose[] ProcessTensorToResult(YoloRawOutput output, Size size)
    {
        int xPadding;
        int yPadding;

        var xRatio = (float)size.Width / metadata.ImageSize.Width;
        var yRatio = (float)size.Height / metadata.ImageSize.Height;

        if (configuration.KeepAspectRatio)
        {
            var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)size.Width, metadata.ImageSize.Height / (float)size.Height);

            xPadding = (int)((metadata.ImageSize.Width - size.Width * reductionRatio) / 2);
            yPadding = (int)((metadata.ImageSize.Height - size.Height * reductionRatio) / 2);

            var maxRatio = Math.Max(xRatio, yRatio);

            xRatio = maxRatio;
            yRatio = maxRatio;
        }
        else
        {
            xPadding = 0;
            yPadding = 0;
        }

        return ProcessTensorToResult(output.Output0, new Vector<int>(xPadding, yPadding), new Vector<float>(xRatio, yRatio));
    }

    private Pose[] ProcessTensorToResult(DenseTensor<float> tensor, Vector<int> padding, Vector<float> ratio)
    {
        var boxes = rawBoundingBoxParser.Parse<RawBoundingBox>(tensor, padding, ratio);

        var shape = metadata.KeypointShape;
        var result = new Pose[boxes.Length];

        var tensorSpan = tensor.Buffer.Span;
        var boxInfoStride = tensor.Strides[1];

        for (var i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];
            var keypoints = new Keypoint[shape.Count];

            for (var index = 0; index < shape.Count; index++)
            {
                var offset = index * shape.Channels + 4 + metadata.Names.Length;

                var pointX = (int)((tensorSpan[offset * boxInfoStride + box.Index] - padding.X) * ratio.X);
                var pointY = (int)((tensorSpan[(offset + 1) * boxInfoStride + box.Index] - padding.Y) * ratio.Y);

                var pointConfidence = metadata.KeypointShape.Channels switch
                {
                    2 => 1f,
                    3 => tensorSpan[(offset + 2) * boxInfoStride + box.Index],
                    _ => throw new InvalidOperationException("Unexpected keypoint shape")
                };

                keypoints[index] = new Keypoint
                {
                    Index = index,
                    Point = new Point(pointX, pointY),
                    Confidence = pointConfidence
                };
            }

            result[i] = new Pose(keypoints)
            {
                Name = box.Name,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };
        }

        return result;
    }
}