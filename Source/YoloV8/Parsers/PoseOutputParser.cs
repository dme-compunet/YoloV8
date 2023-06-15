using Microsoft.ML.OnnxRuntime.Tensors;

using Compunet.YoloV8.Data;
using Compunet.YoloV8.Metadata;
using Compunet.YoloV8.Extensions;

namespace Compunet.YoloV8.Parsers;

internal class PoseOutputParser
{
    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    public PoseOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
    {
        _metadata = metadata;
        _parameters = parameters;
    }

    public IReadOnlyList<IPose> Parse(Tensor<float> output, Size origin)
    {
        var xRatio = origin.Width / _metadata.ImageSize.Width;
        var yRatio = origin.Height / _metadata.ImageSize.Height;

        var poses = new List<Pose>();

        for (int i = 0; i < output.Dimensions[2]; i++)
        {
            var confidence = output[0, 4, i];

            if (confidence < _parameters.Confidence)
                continue;

            var x = output[0, 0, i];
            var y = output[0, 1, i];
            var w = output[0, 2, i];
            var h = output[0, 3, i];

            var xMin = (x - w / 2) * xRatio;
            var yMin = (y - h / 2) * yRatio;
            var xMax = (x + w / 2) * xRatio;
            var yMax = (y + h / 2) * yRatio;

            xMin = Math.Clamp(xMin, 0, origin.Width);
            yMin = Math.Clamp(yMin, 0, origin.Height);
            xMax = Math.Clamp(xMax, 0, origin.Width);
            yMax = Math.Clamp(yMax, 0, origin.Height);

            var rectangle = Rectangle.FromLTRB((int)xMin, (int)yMin, (int)xMax, (int)yMax);
            var keypoints = new List<Keypoint>();

            for (int j = 0; j < 17; j++)
            {
                var offset = j * 3 + 5;

                var px = output[0, offset + 0, i] * xRatio;
                var py = output[0, offset + 1, i] * yRatio;
                var pc = output[0, offset + 2, i];

                if (pc < _parameters.Confidence)
                    continue;

                var keypoint = new Keypoint(j, (int)px, (int)py, pc);
                keypoints.Add(keypoint);
            }

            var pose = new Pose(rectangle, confidence, keypoints);
            poses.Add(pose);
        }

        var suppressed = poses.NonMaxSuppression(x => x.Rectangle,
                                                 x => x.Confidence,
                                                 _parameters.IoU);

        return suppressed;
    }
}