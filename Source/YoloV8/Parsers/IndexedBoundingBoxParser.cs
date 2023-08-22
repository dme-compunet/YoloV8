using Compunet.YoloV8.Extensions;
using Compunet.YoloV8.Metadata;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Compunet.YoloV8.Parsers;

internal readonly struct IndexedBoundingBoxParser
{
    private readonly YoloV8Metadata _metadata;
    private readonly YoloV8Parameters _parameters;

    public IndexedBoundingBoxParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
    {
        _metadata = metadata;
        _parameters = parameters;
    }

    public IReadOnlyList<IndexedBoundingBox> Parse(Tensor<float> output, Size originSize)
    {
        var reductionRatio = Math.Min(_metadata.ImageSize.Width / (float)originSize.Width, _metadata.ImageSize.Height / (float)originSize.Height);

        var xPadding = (int)((_metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2);
        var yPadding = (int)((_metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2);

        return Parse(output, originSize, xPadding, yPadding);
    }

    public IReadOnlyList<IndexedBoundingBox> Parse(Tensor<float> output, Size originSize, int xPadding, int yPadding)
    {
        var metadata = _metadata;
        var parameters = _parameters;

        var magnificationRatio = Math.Max((float)originSize.Width / metadata.ImageSize.Width,
                                          (float)originSize.Height / metadata.ImageSize.Height);

        var boxes = new List<IndexedBoundingBox>(output.Dimensions[2]);

        Parallel.For(0, output.Dimensions[2], i =>
        {
            for (int j = 0; j < metadata.Classes.Count; j++)
            {
                var confidence = output[0, j + 4, i];

                if (confidence <= parameters.Confidence)
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

                var bounds = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);
                var name = metadata.Classes[j];

                var box = new IndexedBoundingBox(i, name, bounds, confidence);
                boxes.Add(box);
            }
        });

        var selected = boxes.NonMaxSuppression(x => x.Bounds,
                                               x => x.Confidence,
                                               _parameters.IoU);

        return selected;
    }
}