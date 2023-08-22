namespace Compunet.YoloV8.Timing;

public readonly struct SpeedResult
{
    public TimeSpan Preprocess { get; }

    public TimeSpan Inference { get; }

    public TimeSpan Postprocess { get; }

    public SpeedResult(TimeSpan preprocess,
                       TimeSpan inference,
                       TimeSpan postprocess)
    {
        Preprocess = preprocess;
        Inference = inference;
        Postprocess = postprocess;
    }

    public override string ToString()
    {
        return $"Preprocess: {Preprocess.TotalSeconds},\tInference: {Inference.TotalSeconds},\tPostprocess: {Postprocess.TotalSeconds}";
    }
}