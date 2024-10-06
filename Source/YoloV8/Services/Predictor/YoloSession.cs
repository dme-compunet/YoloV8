namespace Compunet.YoloV8.Services;

internal class YoloSession(YoloMetadata metadata, InferenceSession session, SessionIoShapeInfo shapeInfo)
{
    public YoloMetadata Metadata => metadata;

    public InferenceSession Session => session;

    public SessionIoShapeInfo ShapeInfo => shapeInfo;
}