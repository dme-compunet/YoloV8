namespace Compunet.YoloV8;

internal readonly ref struct ObbDetectionOutputParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
{
    public ObbBoundingBox[] Parse(Tensor<float> output, Size originSize)
    {
        var boxes = new ObbIndexedBoundingBoxParser(metadata, configuration).Parse(output, originSize);

        var result = new ObbBoundingBox[boxes.Length];

        for (int i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];

            result[i] = new ObbBoundingBox
            {
                Class = box.Class,
                Bounds = box.Bounds,
                Angle = box.Angle,
                Confidence = box.Confidence,
            };
        }

        return result;
    }
}