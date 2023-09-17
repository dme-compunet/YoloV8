namespace Compunet.YoloV8;

public static class YoloV8Extensions
{
    public static IPoseResult Pose(this YoloV8 predictor, ImageSelector selector)
    {
        predictor.CheckPoseShape();

        return predictor.Run(selector, (outputs, image, timer) =>
        {
            var output = outputs[0].AsTensor<float>();

            var parser = new PoseOutputParser(predictor.Metadata, predictor.Parameters);

            var poses = parser.Parse(output, image);

            var speed = timer.Stop();

            return new PoseResult(image, speed, poses);
        });
    }

    public static IDetectionResult Detect(this YoloV8 predictor, ImageSelector selector)
    {
        predictor.EnsureTask(YoloV8Task.Detect);

        return predictor.Run(selector, (outputs, image, timer) =>
        {
            var output = outputs[0].AsTensor<float>();

            var parser = new DetectionOutputParser(predictor.Metadata, predictor.Parameters);

            var boxes = parser.Parse(output, image);

            var speed = timer.Stop();

            return new DetectionResult(image, speed, boxes);
        });
    }

    public static ISegmentationResult Segment(this YoloV8 predictor, ImageSelector selector)
    {
        predictor.EnsureTask(YoloV8Task.Segment);

        return predictor.Run(selector, (outputs, image, timer) =>
        {
            var list = outputs.Select(x => x.AsTensor<float>()).ToList();

            var parser = new SegmentationOutputParser(predictor.Metadata, predictor.Parameters);

            var boxes = parser.Parse(list, image);

            var speed = timer.Stop();

            return new SegmentationResult(image, speed, boxes);
        });
    }

    public static IClassificationResult Classify(this YoloV8 predictor, ImageSelector selector)
    {
        predictor.EnsureTask(YoloV8Task.Classify);

        return predictor.Run<IClassificationResult>(selector, (outputs, image, timer) =>
        {
            var output = outputs[0].AsEnumerable<float>().ToList();

            var probs = new List<(YoloV8Class Class, float Confidence)>(output.Count);

            for (int i = 0; i < output.Count; i++)
            {
                var cls = predictor.Metadata.Classes[i];
                var scr = output[i];

                var prob = (Class: cls, Confidence: scr);

                probs.Add(prob);
            }

            var speed = timer.Stop();

            return new ClassificationResult(image, speed, probs);
        });

    }

    #region Async Operations

    public static async Task<IPoseResult> PoseAsync(this YoloV8 predictor, ImageSelector selector)
    {
        return await Task.Run(() => predictor.Pose(selector));
    }

    public static async Task<IDetectionResult> DetectAsync(this YoloV8 predictor, ImageSelector selector)
    {
        return await Task.Run(() => predictor.Detect(selector));
    }

    public static async Task<ISegmentationResult> SegmentAsync(this YoloV8 predictor, ImageSelector selector)
    {
        return await Task.Run(() => predictor.Segment(selector));
    }

    public static async Task<IClassificationResult> ClassifyAsync(this YoloV8 predictor, ImageSelector selector)
    {
        return await Task.Run(() => predictor.Classify(selector));
    }

    #endregion

    private static void EnsureTask(this YoloV8 predictor, YoloV8Task task)
    {
        if (predictor.Metadata.Task != task)
            throw new InvalidOperationException("The loaded model does not support this task");
    }

    private static void CheckPoseShape(this YoloV8 predictor)
    {
        predictor.EnsureTask(YoloV8Task.Pose);

        if (predictor.Metadata is YoloV8PoseMetadata metadata)
        {
            var shape = metadata.KeypointShape;

            if (shape.Channels is 2 or 3)
                return;
        }

        throw new NotSupportedException("The this keypoint shape is not supported");
    }
}