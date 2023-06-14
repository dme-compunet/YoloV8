using Microsoft.ML.OnnxRuntime;

using Compunet.YoloV8.Timing;

namespace Compunet.YoloV8;

internal delegate TResult PostprocessContext<TResult>(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
                                                      Size image,
                                                      SpeedTimer timer);