namespace Compunet.YoloV8.Parsers;

internal readonly struct DetectionOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
{
    private readonly YoloV8Metadata _metadata = metadata;
    private readonly YoloV8Parameters _parameters = parameters;

    public IEnumerable<BoundingBox> Parse(Tensor<float> output, Size originSize)
    {
        var boxes = new IndexedBoundingBoxParser(_metadata, _parameters).Parse(output, originSize);

        return boxes.AsParallel().Select(box =>
        {
            return new BoundingBox
            {
                Class = box.Class,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };
        });
    }
}