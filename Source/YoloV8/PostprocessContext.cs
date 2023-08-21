using Compunet.YoloV8.Timing;
using Microsoft.ML.OnnxRuntime;

namespace Compunet.YoloV8;

public delegate TResult PostprocessContext<TResult>(IReadOnlyList<NamedOnnxValue> outputs,
                                                    Size imageSize,
                                                    SpeedTimer timer);