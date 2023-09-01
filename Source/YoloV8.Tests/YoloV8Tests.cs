namespace YoloV8.Tests;

public class YoloV8Tests
{
    [Theory]
    [InlineData("bus.jpg", "person:4;bus:1")]
    [InlineData("demo.jpg", "person:2;sports ball:1;baseball bat:1;baseball glove:1")]
    public void DetectionTest(string image, string objects)
    {
        var predictor = Predictors.GetPredictor(YoloV8Task.Detect);

        image = Path.Combine("./assets/input", image);

        var result = predictor.Detect(image);

        var list = new List<(string name, int count)>();

        foreach (var item in objects.Split(';'))
        {
            var split = item.Split(":");

            var name = split[0];
            var count = int.Parse(split[1]);

            list.Add((name, count));
        }

        Assert.Equal(result.Boxes.Count, list.Sum(x => x.count));

        foreach (var (name, count) in list)
        {
            Assert.Equal(result.Boxes.Where(x => x.Class.Name == name).Count(), count);
        }
    }

    [Theory]
    [InlineData("pizza.jpg", "pizza")]
    [InlineData("teddy.jpg", "teddy")]
    [InlineData("toaster.jpg", "toaster")]
    public void ClassificationTest(string image, string label)
    {
        var predictor = Predictors.GetPredictor(YoloV8Task.Classify);

        image = Path.Combine("./assets/input", image);

        var result = predictor.Classify(image);

        Assert.Equal(result.Class.Name, label);
    }

    [Theory]
    [InlineData(YoloV8Task.Pose)]
    [InlineData(YoloV8Task.Detect)]
    [InlineData(YoloV8Task.Segment)]
    [InlineData(YoloV8Task.Classify)]
    public void PredictorTaskTest(YoloV8Task task)
    {
        var predictor = Predictors.GetPredictor(task);

        Assert.Equal(predictor.Metadata.Task, task);
    }
}