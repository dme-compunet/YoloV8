namespace Compunet.YoloSharp.Parsing.Parsers;

internal class DetectionParser(YoloMetadata metadata,
                               IImageAdjustmentService imageAdjustment,
                               IRawBoundingBoxParser rawBoundingBoxParser) : IParser<Detection>
{
    public Detection[] ProcessTensorToResult(IYoloRawOutput output, Size size)
    {
        var adjustment = imageAdjustment.Calculate(size);
        var boxes = rawBoundingBoxParser.Parse<RawBoundingBox>(output.Output0);

        var result = new Detection[boxes.Length];

        for (var i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];

            result[i] = new Detection
            {
                Name = metadata.Names[box.NameIndex],
                Bounds = imageAdjustment.Adjust(box.Bounds, adjustment),
                Confidence = box.Confidence,
            };
        }

        return result;
    }
}