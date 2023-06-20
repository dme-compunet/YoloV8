using Microsoft.ML.OnnxRuntime.Tensors;

using Compunet.YoloV8.Data;
using Compunet.YoloV8.Metadata;
using Compunet.YoloV8.Extensions;

namespace Compunet.YoloV8.Parsers;

internal class DetectionOutputParser
{
    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    public DetectionOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
    {
        _metadata = metadata;
        _parameters = parameters;
    }

    public IReadOnlyList<IBoundingBox> Parse(Tensor<float> output, Size origin)
    {
        var xRatio = (float)origin.Width / _metadata.ImageSize.Width;
        var yRatio = (float)origin.Height / _metadata.ImageSize.Height;

        var boxes = new List<BoundingBox>();

        Parallel.For(0, output.Dimensions[2], i =>
        {
            Parallel.For(4, output.Dimensions[1], j =>
            {
                var confidence = output[0, j, i];

                if (confidence <= _parameters.Confidence)
                    return;

                var x = output[0, 0, i];
                var y = output[0, 1, i];
                var w = output[0, 2, i];
                var h = output[0, 3, i];

                var xMin = (x - w / 2) * xRatio;
                var yMin = (y - h / 2) * yRatio;
                var xMax = (x + w / 2) * xRatio;
                var yMax = (y + h / 2) * yRatio;

                xMin = float.Clamp(xMin, 0, origin.Width);
                yMin = float.Clamp(yMin, 0, origin.Height);
                xMax = float.Clamp(xMax, 0, origin.Width);
                yMax = float.Clamp(yMax, 0, origin.Height);

                var rectangle = Rectangle.FromLTRB((int)xMin, (int)yMin, (int)xMax, (int)yMax);
                var cls = _metadata.Classes[j - 4];

                var box = new BoundingBox(cls, rectangle, confidence);
                boxes.Add(box);
            });
        });

        var selected = boxes.NonMaxSuppression(x => x.Rectangle,
                                               x => x.Confidence,
                                               _parameters.IoU);

        return selected;
    }
}