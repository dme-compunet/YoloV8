namespace Compunet.YoloV8.Parsers;

internal readonly struct DetectionOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
{
    public BoundingBox[] Parse(Tensor<float> output, Size originSize)
    {
        var boxes = new IndexedBoundingBoxParser(metadata, parameters).Parse(output, originSize);

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