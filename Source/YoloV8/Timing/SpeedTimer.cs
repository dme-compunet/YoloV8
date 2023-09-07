namespace Compunet.YoloV8.Timing;

public class SpeedTimer
{
    private readonly Stopwatch _stopwatch;

    private TimeSpan _preprocess;
    private TimeSpan _inference;
    private TimeSpan _postprocess;

    public TimeSpan Preprocess => _preprocess;

    public TimeSpan Inference => _inference;

    public TimeSpan Postprocess => _postprocess;

    public SpeedTimer()
    {
        _stopwatch = new Stopwatch();
    }

    public void StartPreprocess()
    {
        _stopwatch.Restart();
    }

    public void StartInference()
    {
        _preprocess = _stopwatch.Elapsed;
        _stopwatch.Restart();
    }

    public void StartPostprocess()
    {
        _inference = _stopwatch.Elapsed;
        _stopwatch.Restart();
    }

    public SpeedResult Stop()
    {
        _postprocess = _stopwatch.Elapsed;
        _stopwatch.Stop();

        return new SpeedResult(_preprocess,
                               _inference,
                               _postprocess);
    }
}