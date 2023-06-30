using Compunet.YoloV8.Timing;

namespace Compunet.YoloV8.Data;

public interface IYoloV8Result
{
    Size Image { get; }

    SpeedResult Speed { get; }
}