namespace Compunet.YoloV8.Metadata;

internal class SessionIoShapeInfo
{
    public TensorShape Input0 { get; }

    public TensorShape Output0 { get; }

    public TensorShape? Output1 { get; }

    public bool IsDynamicOutput
    {
        get
        {
            if (Output0.IsDynamic)
            {
                return true;
            }

            if (Output1 != null && Output1.Value.IsDynamic)
            {
                return true;
            }

            return false;
        }
    }

    public SessionIoShapeInfo(InferenceSession session)
    {
        var inputMetadata = session.InputMetadata.Values;
        var outputMetadata = session.OutputMetadata.Values;

        Input0 = new TensorShape(inputMetadata.First().Dimensions);
        Output0 = new TensorShape(outputMetadata.First().Dimensions);

        if (session.OutputMetadata.Count == 2)
        {
            Output1 = new TensorShape(outputMetadata.Last().Dimensions);
        }
    }
}