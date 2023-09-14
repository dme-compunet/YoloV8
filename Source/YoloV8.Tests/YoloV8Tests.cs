namespace YoloV8.Tests;

public class YoloV8Tests
{
    [Theory]
    [InlineData("bus.jpg", 4)]
    [InlineData("sports.jpg", 3)]
    public void PoseTest(string image, int count)
    {
        var predictor = Predictors.GetPredictor(YoloV8Task.Pose);

        image = GetImagePath(image);

        var result = predictor.Pose(image);

        Assert.Equal(count, result.Boxes.Count);
    }

    [Theory]
    [InlineData("bus.jpg", "person:4;bus:1")]
    [InlineData("sports.jpg", "person:2;sports ball:1;baseball bat:1;baseball glove:1")]
    public void DetectionTest(string image, string objects)
    {
        var predictor = Predictors.GetPredictor(YoloV8Task.Detect);

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

        Assert.Equal(list.Sum(x => x.count), result.Boxes.Count);

        foreach (var (name, count) in list)
        {
            Assert.Equal(count, result.Boxes.Where(x => x.Class.Name == name).Count());
        }
    }

    [Theory]
    [InlineData("pizza.jpg", "pizza")]
    [InlineData("teddy.jpg", "teddy")]
    [InlineData("toaster.jpg", "toaster")]
    public void ClassificationTest(string image, string label)
    {
        var predictor = Predictors.GetPredictor(YoloV8Task.Classify);

        image = GetImagePath(image);

        var result = predictor.Classify(image);

        Assert.Equal(result.Class.Name, label);
    }

    [Theory]
    [InlineData(YoloV8Task.Pose, 1, 640)]
    [InlineData(YoloV8Task.Detect, 80, 640)]
    [InlineData(YoloV8Task.Segment, 80, 640)]
    [InlineData(YoloV8Task.Classify, 1000, 224)]
    public void MetadataTest(YoloV8Task task, int classesCount, int imageSize)
    {
        var metadata = Predictors.GetPredictor(task).Metadata;

        Assert.Equal("Ultralytics", metadata.Author);
        Assert.Contains("Ultralytics YOLOv8", metadata.Description);
        Assert.StartsWith("8.0", metadata.Version);

        Assert.Equal(task, metadata.Task);

        Assert.Equal(imageSize, metadata.ImageSize.Width);
        Assert.Equal(imageSize, metadata.ImageSize.Height);

        Assert.Equal(classesCount, metadata.Classes.Count);
    }

    private static string GetImagePath(string image)
    {
        return Path.Combine("./assets/input", image);
    }
}