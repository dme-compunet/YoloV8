namespace Compunet.YoloV8.Services;

internal class SessionRunnerService(InferenceSession session,
                                    SessionTensorInfo tensorInfo,
                                    YoloConfiguration configuration,
                                    YoloMetadata metadata,
                                    IPreprocessService preprocess,
                                    IMemoryAllocatorService memoryAllocator) : ISessionRunnerService
{
    private readonly object _lock = new();
    private readonly RunOptions _options = new();

    public YoloRawOutput PreprocessAndRun(Image<Rgb24> image, out PredictorTimer timer)
    {
        // Create timer
        timer = new PredictorTimer();

        // Create io binding
        using var binding = session.CreateIoBinding();

        // Create and bind raw output
        var output = CreateRawOutput(binding);

        // Start pre-process timer
        timer.StartPreprocess();

        // Allocate the input tensor
        using var input = memoryAllocator.AllocateTensor<float>(tensorInfo.Input0, true);

        // Preprocess image to tensor and bind to ort binding
        ProcessInput(image, input.Tensor, binding);

        // Start inference timer
        timer.StartInference();

        // Run the model
        if (configuration.SuppressParallelInference)
        {
            lock (_lock)
            {
                session.RunWithBinding(_options, binding);
            }
        }
        else
        {
            session.RunWithBinding(_options, binding);
        }

        // Return the yolo raw output
        return output;
    }

    private YoloRawOutput CreateRawOutput(OrtIoBinding binding)
    {
        var output0Info = tensorInfo.Output0;
        var output1Info = tensorInfo.Output1;

        // Allocate output0 tensor buffer
        var output0 = memoryAllocator.AllocateTensor<float>(output0Info);

        // Bind tensor buffer to ort binding
        binding.BindOutput(session.OutputNames[0], CreateOrtValue(output0.Tensor.Buffer, output0Info.Dimensions64));

        if (output1Info != null)
        {
            // Allocate output1 tensor buffer
            var output1 = memoryAllocator.AllocateTensor<float>(output1Info.Value);

            // Bind tensor buffer to ort binding
            binding.BindOutput(session.OutputNames[1], CreateOrtValue(output1.Tensor.Buffer, output1Info.Value.Dimensions64));

            return new YoloRawOutput(output0, output1);
        }

        return new YoloRawOutput(output0, null);
    }

    #region Preprocess

    private void ProcessInput(Image<Rgb24> image, DenseTensor<float> target, OrtIoBinding binding)
    {
        // Apply auto orient if required
        if (configuration.SkipImageAutoOrient == false)
        {
            image.AutoOrient();
        }

        // Resize the input image
        using var resized = ResizeImage(image, out var padding);

        // Process the image to tensor
        preprocess.ProcessImageToTensor(resized, target, padding);

        // Create ort values
        var ortInput = CreateOrtValue(target.Buffer, tensorInfo.Input0.Dimensions64);

        // Bind input to ort io binding
        binding.BindInput(session.InputNames[0], ortInput);
    }

    private Image<Rgb24> ResizeImage(Image<Rgb24> image, out Vector<int> padding)
    {
        // Get the model image input size
        var inputSize = metadata.ImageSize;

        // Create resize options
        var options = new ResizeOptions()
        {
            Size = inputSize,

            // Select resize mode according to 'keepAspectRatio'
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