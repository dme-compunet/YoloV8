namespace Compunet.YoloV8;

public interface IYoloV8Builder
{
    public IYoloV8Builder UseOnnxModel(BinarySelector model);

#if GPURELEASE

   public IYoloV8Builder UseCuda(int deviceId = 0);
   public IYoloV8Builder UseCuda(OrtCUDAProviderOptions options);

   public IYoloV8Builder UseRocm(int deviceId = 0);
   public IYoloV8Builder UseRocm(OrtROCMProviderOptions options);

   public IYoloV8Builder UseTensorrt(int deviceId = 0);
   public IYoloV8Builder UseTensorrt(OrtTensorRTProviderOptions options);

   public IYoloV8Builder UseTvm(string settings = "");

#endif

   public IYoloV8Builder WithMetadata(YoloV8Metadata metadata);

    public IYoloV8Builder WithConfiguration(Action<YoloV8Configuration> configure);

    public IYoloV8Builder WithSessionOptions(SessionOptions sessionOptions);

    public YoloV8Predictor Build();
}