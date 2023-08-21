using Compunet.YoloV8.Data;
using Compunet.YoloV8.Extensions;
using Compunet.YoloV8.Metadata;
using Microsoft.ML.OnnxRuntime.Tensors;

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

        var xPadding = (metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2;
        var yPadding = (metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2;

        var magnificationRatio = Math.Max((float)originSize.Width / metadata.ImageSize.Width, (float)originSize.Height / metadata.ImageSize.Height);

        var boxes = new List<PoseBoundingBox>(output.Dimensions[2]);

        var shape = metadata.KeypointShape;

        Parallel.For(0, output.Dimensions[2], i =>
        {
            for (int j = 0; j < metadata.Classes.Count; j++)
            {
                var confidence = output[0, j + 4, i];

                if (confidence < parameters.Confidence)
                    continue;

                var x = output[0, 0, i];
                var y = output[0, 1, i];
                var w = output[0, 2, i];
                var h = output[0, 3, i];

                var xMin = (int)((x - w / 2 - xPadding) * magnificationRatio);
                var yMin = (int)((y - h / 2 - yPadding) * magnificationRatio);
                var xMax = (int)((x + w / 2 - xPadding) * magnificationRatio);
                var yMax = (int)((y + h / 2 - yPadding) * magnificationRatio);

                xMin = Math.Clamp(xMin, 0, originSize.Width);
                yMin = Math.Clamp(yMin, 0, originSize.Height);
                xMax = Math.Clamp(xMax, 0, originSize.Width);
                yMax = Math.Clamp(yMax, 0, originSize.Height);

                var rectangle = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);
                var name = metadata.Classes[j];

                var keypoints = new List<Keypoint>();

                for (int k = 0; k < shape.Count; k++)
                {
                    var offset = k * shape.Channels + 4 + metadata.Classes.Count;

                    var pointX = (int)((output[0, offset + 0, i] - xPadding) * magnificationRatio);
                    var pointY = (int)((output[0, offset + 1, i] - yPadding) * magnificationRatio);

                    var pointConfidence = metadata.KeypointShape.Channels switch
                    {
                        2 => 1F,
                        3 => output[0, offset + 2, i],
                        _ => throw new NotSupportedException("Unexpected keypoint shape")
                    };

                    var keypoint = new Keypoint(k, pointX, pointY, pointConfidence);
                    keypoints.Add(keypoint);
                }

                var box = new PoseBoundingBox(name, rectangle, confidence, keypoints);
                boxes.Add(box);
            }
        });

        var selected = boxes.NonMaxSuppression(x => x.Rectangle,
                                               x => x.Confidence,
                                               parameters.IoU);

        return selected;
    }
}