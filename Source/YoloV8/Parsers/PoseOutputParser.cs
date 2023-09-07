namespace Compunet.YoloV8.Parsers;

internal readonly struct PoseOutputParser
{
    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    public PoseOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
    {
        _metadata = metadata;
        _parameters = parameters;
    }

    public IReadOnlyList<IPoseBoundingBox> Parse(Tensor<float> output, Size originSize)
    {
        var metadata = (YoloV8PoseMetadata)_metadata;
        var parameters = _parameters;

        var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)originSize.Width, metadata.ImageSize.Height / (float)originSize.Height);

        var xPadding = (int)((metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2);
        var yPadding = (int)((metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2);

        var magnificationRatio = Math.Max((float)originSize.Width / metadata.ImageSize.Width,
                                          (float)originSize.Height / metadata.ImageSize.Height);

        var boxes = new IndexedBoundingBoxParser(_metadata, _parameters).Parse(output, originSize, xPadding, yPadding);

        var shape = metadata.KeypointShape;

        return boxes.SelectParallely(box =>
        {
            var keypoints = new Keypoint[shape.Count];

            for (int i = 0; i < shape.Count; i++)
            {
                var offset = i * shape.Channels + 4 + metadata.Classes.Count;

                var pointX = (int)((output[0, offset + 0, box.Index] - xPadding) * magnificationRatio);
                var pointY = (int)((output[0, offset + 1, box.Index] - yPadding) * magnificationRatio);

                var pointConfidence = metadata.KeypointShape.Channels switch
                {
                    2 => 1F,
                    3 => output[0, offset + 2, box.Index],
                    _ => throw new NotSupportedException("Unexpected keypoint shape")
                };

                keypoints[i] = new Keypoint(i, pointX, pointY, pointConfidence);
            }

            return new PoseBoundingBox(box.Class, box.Bounds, box.Confidence, keypoints);
        });
    }
}