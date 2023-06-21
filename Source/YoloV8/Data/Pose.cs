namespace Compunet.YoloV8.Data;

internal class Pose : IPose
{
    public Rectangle Rectangle { get; }

    public float Confidence { get; }

    public IReadOnlyList<IKeypoint> Keypoints { get; }

    public Pose(Rectangle rectangle, float confidence, IReadOnlyList<Keypoint> keypoints)
    {
        Rectangle = rectangle;
        Confidence = confidence;
        Keypoints = keypoints;
    }

    public IKeypoint? GetKeypoint(int id)
    {
        return Keypoints.SingleOrDefault(x => x.Id == id);
    }
}