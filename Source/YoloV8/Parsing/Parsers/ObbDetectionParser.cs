namespace Compunet.YoloV8.Parsing.Parsers;

internal class ObbDetectionParser(YoloMetadata metadata,
                                  IImageAdjustmentService imageAdjustment,
                                  IRawBoundingBoxParser rawBoundingBoxParser) : IParser<ObbDetection>
{
    public ObbDetection[] ProcessTensorToResult(IYoloRawOutput output, Size size)
    {
        var adjustment = imageAdjustment.Calculate(size);
        var boxes = rawBoundingBoxParser.Parse<RawObbBoundingBox>(output.Output0);

        var result = new ObbDetection[boxes.Length];

        for (var i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];

            result[i] = new ObbDetection
            {
                Name = metadata.Names[box.NameIndex],
                Angle = box.Angle,
                Bounds = imageAdjustment.Adjust(box.Bounds, adjustment),
                Confidence = box.Confidence,
            };
        }

        return result;
    }
}