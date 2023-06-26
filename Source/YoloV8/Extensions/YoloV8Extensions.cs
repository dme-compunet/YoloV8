using Compunet.YoloV8.Data;
using Compunet.YoloV8.Parsers;
using Compunet.YoloV8.Metadata;

namespace Compunet.YoloV8.Extensions;

public static class YoloV8Extensions
{
    public static IPoseResult Pose(this YoloV8 predictor, ImageSelector selector)
    {
        predictor.EnsurePoseShape();

        return predictor.Run(selector, (outputs, image, timer) =>
        {
            var output = outputs[0].AsTensor<float>();

            var parser = new PoseOutputParser(predictor.Metadata, predictor.Parameters);

            var poses = parser.Parse(output, image);

            var speed = timer.Stop();

            return new PoseResult(image,
                                  speed,
                                  poses);
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

            return new DetectionResult(image,
                                       speed,
                                       boxes);
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

    private static void EnsureTask(this YoloV8 predictor, YoloV8Task task)
    {
        if (predictor.Metadata.Task != task)
            throw new InvalidOperationException("The loaded model does not support this task");
    }

    private static void EnsurePoseShape(this YoloV8 predictor)
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
