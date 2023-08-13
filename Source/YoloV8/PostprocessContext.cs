using Microsoft.ML.OnnxRuntime;

using Compunet.YoloV8.Timing;

namespace Compunet.YoloV8;

public delegate TResult PostprocessContext<TResult>(IReadOnlyList<NamedOnnxValue> outputs,
                                                    Size imageSize,
                                                    SpeedTimer timer);