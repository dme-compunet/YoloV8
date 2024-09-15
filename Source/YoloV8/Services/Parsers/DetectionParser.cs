namespace Compunet.YoloV8.Services;

internal class DetectionParser(IRawBoundingBoxParser rawBoundingBoxParser) : IParser<Detection>
{
    public Detection[] ProcessTensorToResult(YoloRawOutput output, Size size)
    {
        var boxes = rawBoundingBoxParser.Parse<RawBoundingBox>(output.Output0, size);

        var result = new Detection[boxes.Length];

        for (var i = 0; i < boxes.Length; i++)
        {
            var box = boxes[i];

            result[i] = new Detection
            {
                Name = box.Name,
                Bounds = box.Bounds,
                Confidence = box.Confidence,
            };
        }

        return result;
    }
}