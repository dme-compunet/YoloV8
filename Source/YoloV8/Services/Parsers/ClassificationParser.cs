namespace Compunet.YoloV8.Services;

internal class ClassificationParser(YoloMetadata metadata) : IParser<Classification>
{
    public Classification[] ProcessTensorToResult(YoloRawOutput tensor, Size size)
    {
        var tensorSpan = tensor.Output0.Buffer.Span;

        var result = new Classification[tensorSpan.Length];

        for (var i = 0; i < tensorSpan.Length; i++)
        {
            var name = metadata.Names[i];
            var confidence = tensorSpan[i];

            result[i] = new Classification
            {
                Name = name,
                Confidence = confidence,
            };
        }

        result.AsSpan().Sort((x, y) => y.Confidence.CompareTo(x.Confidence));

        return result;
    }
}