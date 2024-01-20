namespace Compunet.YoloV8.Data;

public class PoseBoundingBox : BoundingBox
{
    public required Keypoint[] Keypoints { get; init; }

    public Keypoint? GetKeypoint(int index)
    {
        return Keypoints.SingleOrDefault(x => x.Index == index);
    }
}