namespace Compunet.YoloV8.Services;

internal class ObbDetectionParser(IRawBoundingBoxParser rawBoundingBoxParser) : IParser<ObbDetection>
{
    public ObbDetection[] ProcessTensorToResult(YoloRawOutput output, Size size)
    {
        var boxes = rawBoundingBoxParser.Parse<RawObbBoundingBox>(output.Output0, size);

        var result = new ObbDetection[boxes.Length];

        for (var i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];

            result[i] = new ObbDetection
            {
                Name = box.Name,
                Angle = box.Angle,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };
        }

        return result;
    }
}