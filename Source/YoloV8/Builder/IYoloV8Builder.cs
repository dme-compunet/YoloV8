namespace Compunet.YoloV8;

public interface IYoloV8Builder
{
    //public IYoloV8Builder UseMetadateDetect();

    public IYoloV8Builder UseOnnxModel(BinarySelector model);

    public IYoloV8Builder UseCuda() => UseCuda(0);

    public IYoloV8Builder UseCuda(int deviceId);

    //public IYoloV8Builder UseDefaultConfiguration();

    public IYoloV8Builder WithMetadata(YoloV8Metadata metadata);

    public IYoloV8Builder WithConfiguration(Action<YoloV8Configuration> configure);

    public YoloV8Predictor Build();
}