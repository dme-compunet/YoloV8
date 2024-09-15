namespace YoloV8.Tests;

public class YoloPredictorTests
{
    [Theory]
    [InlineData("bus.jpg", 3)]
    [InlineData("sports.jpg", 3)]
    public void PoseTest(string image, int count)
    {
        var predictor = Predictors.GetPredictor(YoloTask.Pose);

        image = GetImagePath(image);

        var result = predictor.Pose(image);

        Assert.Equal(count, result.Count);
    }

    [Theory]
    [InlineData("bus.jpg", "person:3;bus:1")]
    [InlineData("sports.jpg", "person:2;sports ball:1;baseball bat:1;baseball glove:2")]
    public void DetectionTest(string image, string objects)
    {
        var predictor = Predictors.GetPredictor(YoloTask.Detect);

        image = GetImagePath(image);

        var result = predictor.Detect(image);

        var list = new List<(string name, int count)>();

        foreach (var item in objects.Split(';'))
        {
            var split = item.Split(":");

            var name = split[0];
            var count = int.Parse(split[1]);

            list.Add((name, count));
        }

        Assert.Equal(list.Sum(x => x.count), result.Count);

        foreach (var (name, count) in list)
        {
            Assert.Equal(count, result.Where(x => x.Name.Name == name).Count());
        }
    }

    [Theory]
    [InlineData("pizza.jpg", "pizza")]
    [InlineData("teddy.jpg", "teddy")]
    [InlineData("toaster.jpg", "toaster")]
    public void ClassificationTest(string image, string label)
    {
        var predictor = Predictors.GetPredictor(YoloTask.Classify);

        image = GetImagePath(image);

        var result = predictor.Classify(image);

        Assert.Equal(result[0].Name.Name, label);
    }

    [Theory]
    [InlineData(YoloTask.Pose, 1, 640)]
    [InlineData(YoloTask.Detect, 80, 640)]
    [InlineData(YoloTask.Segment, 80, 640)]
    [InlineData(YoloTask.Classify, 1000, 224)]
    public void MetadataTest(YoloTask task, int classesCount, int imageSize)
    {
        var metadata = Predictors.GetPredictor(task).Metadata;

        Assert.Equal("Ultralytics", metadata.Author);
        Assert.Contains("Ultralytics YOLOv8", metadata.Description);
        Assert.StartsWith("8.", metadata.Version);

        Assert.Equal(task, metadata.Task);

        Assert.Equal(1, metadata.BatchSize);

        Assert.Equal(imageSize, metadata.ImageSize.Width);
        Assert.Equal(imageSize, metadata.ImageSize.Height);

        Assert.Equal(classesCount, metadata.Names.Length);
    }

    private static string GetImagePath(string image)
    {
        return Path.Combine("./images", image);
    }
}