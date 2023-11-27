namespace Compunet.YoloV8.Parsers;

internal readonly struct DetectionOutputParser(YoloV8Metadata metadata, YoloV8Parameters parameters)
{
    private readonly YoloV8Metadata _metadata = metadata;
    private readonly YoloV8Parameters _parameters = parameters;

    public IReadOnlyList<IBoundingBox> Parse(Tensor<float> output, Size originSize)
    {
        var boxes = new IndexedBoundingBoxParser(_metadata, _parameters).Parse(output, originSize);

        return boxes.SelectParallely(CreateBoundingBox);
    }

    private static BoundingBox CreateBoundingBox(IndexedBoundingBox value)
    {
        return new BoundingBox(value.Class, value.Bounds, value.Confidence);
    }
}