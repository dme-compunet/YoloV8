using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Data;

internal class PoseBoundingBox : BoundingBox,  IPoseBoundingBox
{
    public IReadOnlyList<IKeypoint> Keypoints { get; }

    public PoseBoundingBox(YoloV8Class _class,
                           Rectangle rectangle,
                           float confidence,
                           IReadOnlyList<Keypoint> keypoints) : base(_class, rectangle, confidence)
    {
        Keypoints = keypoints;
    }

    public IKeypoint? GetKeypoint(int index)
    {
        return Keypoints.SingleOrDefault(x => x.Index == index);
    }
}