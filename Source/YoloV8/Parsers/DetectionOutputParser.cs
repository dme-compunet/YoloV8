namespace Compunet.YoloV8.Parsers;

internal readonly ref struct DetectionOutputParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
{
    public BoundingBox[] Parse(Tensor<float> output, Size originSize)
    {
        var boxes = new IndexedBoundingBoxParser(metadata, configuration).Parse(output, originSize);

        var result = new BoundingBox[boxes.Length];

        for (int i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];

            result[i] = new BoundingBox
            {
                Class = box.Class,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };
        }

        return result;
    }
}