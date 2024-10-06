namespace Compunet.YoloV8.Services;

internal class SessionRunnerService(YoloSession yoloSession,
                                    YoloConfiguration configuration,
                                    IPixelsNormalizerService normalizer,
                                    IMemoryAllocatorService memoryAllocator) : ISessionRunnerService
{
    private readonly object _lock = new();
    private readonly RunOptions _options = new();

    public InferenceSession Session => yoloSession.Session;

    public SessionIoShapeInfo IoShapeInfo => yoloSession.ShapeInfo;

    public IYoloRawOutput PreprocessAndRun(Image<Rgb24> image, out PredictorTimer timer)
    {
        // Create timer
        timer = new PredictorTimer();

        // Start pre-process timer
        timer.StartPreprocess();

        // Allocate the input tensor
        using var input = memoryAllocator.AllocateTensor<float>(IoShapeInfo.Input0, true);

        // Preprocess image to tensor and bind to ort binding
        NormalizeInput(image, input.Tensor);

        // Start inference timer
        timer.StartInference();

        if (IoShapeInfo.IsDynamicOutput)
        {
            return RunWithDynamicOutput(input.Tensor, ref timer);
        }
        else
        {
            return RunWithFixedOutput(input.Tensor, ref timer);
        }
    }

    private YoloRawOutput RunWithFixedOutput(DenseTensor<float> input, ref PredictorTimer timer)
    {
        // Create io binding
        using var binding = Session.CreateIoBinding();

        // Bind the input
        binding.BindInput(Session.InputNames[0], CreateOrtValue(input.Buffer, IoShapeInfo.Input0.Dimensions64));

        // Create and bind raw output
        var output = CreateRawOutput(binding);

        // Run the model
        if (configuration.SuppressParallelInference)
        {
            lock (_lock)
            {
                Session.RunWithBinding(_options, binding);
            }
        }
        else
        {
            Session.RunWithBinding(_options, binding);
        }

        // Return the yolo raw output
        return output;
    }

    private OrtYoloRawOutput RunWithDynamicOutput(DenseTensor<float> input, ref PredictorTimer timer)
    {
        var inputs = new NamedOnnxValue[]
        {
            NamedOnnxValue.CreateFromTensor(Session.InputNames[0], input)
        };

        if (configuration.SuppressParallelInference)
        {
            lock (_lock)
            {
                return new OrtYoloRawOutput(Session.Run(inputs));
            }
        }
        else
        {
            return new OrtYoloRawOutput(Session.Run(inputs));
        }
    }

    private YoloRawOutput CreateRawOutput(OrtIoBinding binding)
    {
        var output0Info = IoShapeInfo.Output0;
        var output1Info = IoShapeInfo.Output1;

        // Allocate output0 tensor buffer
        var output0 = memoryAllocator.AllocateTensor<float>(output0Info);

        // Bind tensor buffer to ort binding
        binding.BindOutput(Session.OutputNames[0], CreateOrtValue(output0.Tensor.Buffer, output0Info.Dimensions64));

        if (output1Info != null)
        {
            // Allocate output1 tensor buffer
            var output1 = memoryAllocator.AllocateTensor<float>(output1Info.Value);

            // Bind tensor buffer to ort binding
            binding.BindOutput(Session.OutputNames[1], CreateOrtValue(output1.Tensor.Buffer, output1Info.Value.Dimensions64));

            return new YoloRawOutput(output0, output1);
        }

        return new YoloRawOutput(output0, null);
    }

    #region Preprocess

    private void NormalizeInput(Image<Rgb24> image, DenseTensor<float> target)
    {
        // Apply auto orient if required
        if (configuration.ApplyAutoOrient)
        {
            image.AutoOrient();
        }

        // Resize the input image
        using var resized = ResizeImage(image, out var padding);

        // Process the image to tensor
        normalizer.NormalizerPixelsToTensor(resized, target, padding);
    }


    private Image<Rgb24> ResizeImage(Image<Rgb24> image, out Vector<int> padding)
    {
        // Get the model image input size
        var inputSize = yoloSession.Metadata.ImageSize;

        // Create resize options
        var options = new ResizeOptions()
        {
            Size = inputSize,

            // Select resize mode according to 'KeepAspectRatio'
            Mode = configuration.KeepAspectRatio
                   ? ResizeMode.Max
                   : ResizeMode.Stretch,

            // Select faster resampling algorithm
            Sampler = KnownResamplers.NearestNeighbor
        };

        // Create resized image
        var resized = image.Clone(x => x.Resize(options));

        // Calculate padding
        padding = new Vector<int>(
            (inputSize.Width - resized.Size.Width) / 2,
            (inputSize.Height - resized.Size.Height) / 2
        );

        // Return the resized image
        return resized;
    }

    #endregion

    private static OrtValue CreateOrtValue(Memory<float> buffer, long[] shape)
    {
        return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, buffer, shape);
    }
}