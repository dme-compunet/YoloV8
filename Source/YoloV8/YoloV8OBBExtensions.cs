using ClipperLib;
using Compunet.YoloV8.Data;
using Compunet.YoloV8.Metadata;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using System.Text;
using static Compunet.YoloV8.YoloV8Extensions;
//using static Compunet.YoloV8.YoloV8Extensions;

namespace Compunet.YoloV8
{
    public static partial class YoloV8Extensions
    {
        public static async Task<OBBDetectionResult> DetectOBBAsync(this YoloV8Predictor predictor, ImageSelector selector)
        {
            YoloV8Predictor predictor2 = predictor;
            ImageSelector selector2 = selector;
            return await Task.Run(() => predictor2.DetectOBB(selector2));
        }

        public static OBBDetectionResult DetectOBB(this YoloV8Predictor predictor, ImageSelector selector)
        {
            predictor.ValidateTask(YoloV8Task.Obb);

            return predictor.Run(selector, (outputs, image, timer) =>
            {
                var output = outputs[0].AsTensor<float>();

                var parser = new OBBDetectionOutputParser(predictor.Metadata, predictor.Configuration);

                var boxes = parser.Parse(output, image);

                var speed = timer.Stop();

                return new OBBDetectionResult
                {
                    Boxes = boxes,
                    Image = image,
                    Speed = speed,
                };
            });
        }

        internal readonly struct OBBDetectionOutputParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
        {
            public OBBBoundingBox[] Parse(Tensor<float> output, SixLabors.ImageSharp.Size originSize)
            {
                var boxes = new OBBIndexedBoundingBoxParser(metadata, configuration).Parse(output, originSize);

                var result = new OBBBoundingBox[boxes.Length];

                for (int i = 0; i < boxes.Length; i++)
                {
                    var box = boxes[i];

                    result[i] = new OBBBoundingBox
                    {
                        Class = box.Class,
                        X = box.X,
                        Y = box.Y,
                        Width = box.Width,
                        Height = box.Height,
                        Angle = box.Angle,
                        Confidence = box.Confidence,
                    };
                }

                return result;
            }
        }

        internal readonly struct OBBIndexedBoundingBox : IComparable<OBBIndexedBoundingBox>
        {
            public bool IsEmpty => X == default && Y == default && Width == 0 && Height == 0 && Confidence == 0;

            public required int Index { get; init; }

            public required YoloV8Class Class { get; init; }

            public required float X { get; init; }
            public required float Y { get; init; }
            public required float Width { get; init; }
            public required float Height { get; init; }
            public required float Angle { get; init; }

            public required float Confidence { get; init; }

            public int CompareTo(OBBIndexedBoundingBox other) => Confidence.CompareTo(other.Confidence);

            public SixLabors.ImageSharp.Point[] GetCornerPoints()
            {
                // Calculate half-width and half-height
                float halfWidth = Width / 2;
                float halfHeight = Height / 2;

                // Calculate the rotation matrix components
                float cosTheta = (float)Math.Cos(Angle);
                float sinTheta = (float)Math.Sin(Angle);

                // Calculate the coordinates of the four corners
                float[,] cornerOffsets = {
                    { -halfWidth, -halfHeight },
                    { halfWidth, -halfHeight },
                    { halfWidth, halfHeight },
                    { -halfWidth, halfHeight }
                };

                var cornerPoints = new SixLabors.ImageSharp.Point[4];
                for (int i = 0; i < 4; i++)
                {
                    float xOffset = cornerOffsets[i, 0];
                    float yOffset = cornerOffsets[i, 1];

                    // Apply rotation
                    float rotatedX = X + (xOffset * cosTheta - yOffset * sinTheta);
                    float rotatedY = Y + (xOffset * sinTheta + yOffset * cosTheta);

                    cornerPoints[i] = new((int)rotatedX, (int)rotatedY);
                }

                return cornerPoints.OrderPointsClockwiseTopLeft();
            }
        }

        internal readonly struct OBBIndexedBoundingBoxParser(YoloV8Metadata metadata, YoloV8Configuration configuration)
        {
            public OBBIndexedBoundingBox[] Parse(Tensor<float> output, SixLabors.ImageSharp.Size originSize)
            {
                int xPadding;
                int yPadding;

                if (configuration.KeepOriginalAspectRatio)
                {
                    var reductionRatio = Math.Min(metadata.ImageSize.Width / (float)originSize.Width,
                                                  metadata.ImageSize.Height / (float)originSize.Height);

                    xPadding = (int)((metadata.ImageSize.Width - originSize.Width * reductionRatio) / 2);
                    yPadding = (int)((metadata.ImageSize.Height - originSize.Height * reductionRatio) / 2);
                }
                else
                {
                    xPadding = 0;
                    yPadding = 0;
                }

                return Parse(output, originSize, xPadding, yPadding);
            }

            public OBBIndexedBoundingBox[] Parse(Tensor<float> output, SixLabors.ImageSharp.Size originSize, int xPadding, int yPadding)
            {
                var xRatio = (float)originSize.Width / metadata.ImageSize.Width;
                var yRatio = (float)originSize.Height / metadata.ImageSize.Height;

                if (configuration.KeepOriginalAspectRatio)
                {
                    var maxRatio = Math.Max(xRatio, yRatio);

                    xRatio = maxRatio;
                    yRatio = maxRatio;
                }

                return Parse(output, originSize, xPadding, yPadding, xRatio, yRatio);
            }

            public OBBIndexedBoundingBox[] Parse(Tensor<float> output, SixLabors.ImageSharp.Size originSize, int xPadding, int yPadding, float xRatio, float yRatio)
            {
                var _metadata = metadata;
                var _parameters = configuration;

                var detectionDataSize = output.Dimensions[1];
                var boxes = new OBBIndexedBoundingBox[output.Dimensions[2]];

                Parallel.For(0, output.Dimensions[2], i =>
                {
                    var maxConfidence = _parameters.Confidence;
                    var maxConfidenceIndex = -1;
                    for (int j = 0; j < _metadata.Names.Count; j++)
                    {
                        var confidence = output[0, j + 4, i];

                        if (confidence > maxConfidence)
                        {
                            maxConfidence = confidence;
                            maxConfidenceIndex = j;
                        }
                    }
                    if (maxConfidenceIndex == -1)
                        return;

                    var x = output[0, 0, i] * xRatio - xPadding;
                    var y = output[0, 1, i] * yRatio - yPadding;
                    var w = output[0, 2, i] * xRatio - xPadding * 2;
                    var h = output[0, 3, i] * yRatio - yPadding * 2;
                    var a = output[0, detectionDataSize-1, i];

                    var xMin = (int)((x - w / 2 - xPadding) * xRatio);
                    var yMin = (int)((y - h / 2 - yPadding) * yRatio);
                    var xMax = (int)((x + w / 2 - xPadding) * xRatio);
                    var yMax = (int)((y + h / 2 - yPadding) * yRatio);

                    xMin = Math.Clamp(xMin, 0, originSize.Width);
                    yMin = Math.Clamp(yMin, 0, originSize.Height);
                    xMax = Math.Clamp(xMax, 0, originSize.Width);
                    yMax = Math.Clamp(yMax, 0, originSize.Height);

                    var name = _metadata.Names[maxConfidenceIndex];
                    var bounds = Rectangle.FromLTRB(xMin, yMin, xMax, yMax);

                    boxes[i] = new OBBIndexedBoundingBox
                    {
                        Index = i,
                        Class = name,
                        X = x,
                        Y = y,
                        Width = w,
                        Height = h,
                        Angle = a,
                        Confidence = maxConfidence
                    };

                });

                var count = 0;

                for (int i = 0; i < boxes.Length; i++)
                {
                    if (boxes[i].IsEmpty == false)
                    {
                        count++;
                    }
                }

                var topBoxes = new OBBIndexedBoundingBox[count];

                var topIndex = 0;

                for (int i = 0; i < boxes.Length; i++)
                {
                    var box = boxes[i];

                    if (box.IsEmpty)
                    {
                        continue;
                    }

                    topBoxes[topIndex++] = box;
                }

                return OBBNonMaxSuppressionHelper.Suppress(topBoxes, configuration.IoU);
            }
        }
    }
}

namespace Compunet.YoloV8.Data
{

    public class OBBBoundingBox
    {
        public required YoloV8Class Class { get; init; }

        public required float X { get; init; }
        public required float Y { get; init; }
        public required float Width { get; init; }
        public required float Height { get; init; }
        public required float Angle { get; init; }

        public required float Confidence { get; init; }

        public override string ToString()
        {
            return $"{Class.Name} ({Confidence:N})";
        }

        public SixLabors.ImageSharp.Point[] GetCornerPoints()
        {
            // Calculate half-width and half-height
            float halfWidth = Width / 2;
            float halfHeight = Height / 2;

            // Calculate the rotation matrix components
            float cosTheta = (float)Math.Cos(Angle);
            float sinTheta = (float)Math.Sin(Angle);

            // Calculate the coordinates of the four corners
            float[,] cornerOffsets = {
                { -halfWidth, -halfHeight },
                { halfWidth, -halfHeight },
                { halfWidth, halfHeight },
                { -halfWidth, halfHeight }
            };

            var cornerPoints = new SixLabors.ImageSharp.Point[4];
            for (int i = 0; i < 4; i++)
            {
                float xOffset = cornerOffsets[i, 0];
                float yOffset = cornerOffsets[i, 1];

                // Apply rotation
                float rotatedX = X + (xOffset * cosTheta - yOffset * sinTheta);
                float rotatedY = Y + (xOffset * sinTheta + yOffset * cosTheta);

                cornerPoints[i] = new SixLabors.ImageSharp.Point((int)rotatedX, (int)rotatedY);
            }
            return cornerPoints.OrderPointsClockwiseTopLeft();
        }
    }

    public class OBBDetectionResult : YoloV8Result
    {
        public required OBBBoundingBox[] Boxes { get; init; }

        public override string ToString()
        {
            //return Boxes.Summary();
            var result = new StringBuilder();
            for (int i = 0; i < Boxes.Length; i++)
            {
                result.AppendLine(Boxes[i].ToString());
            }
            return result.ToString();
        }
    }
}

internal static class PointsHelper
{
    public static SixLabors.ImageSharp.Point[] OrderPointsClockwiseTopLeft(this SixLabors.ImageSharp.Point[] points)
    {
        if (points.Length <= 2)
        {
            return points; // No need to sort for 0, 1, or 2 points
        }

        points = points.Clone() as SixLabors.ImageSharp.Point[];

        // Find the top-left and bottom-right points
        var topLeft = points[0];
        var bottomRight = points[0];
        for (int i = 1; i < points.Length; i++)
        {
            if (points[i].Y < topLeft.Y ||
                (points[i].Y == topLeft.Y && points[i].X < topLeft.X))
            {
                topLeft = points[i];
            }

            if (points[i].Y > bottomRight.Y ||
                (points[i].Y == bottomRight.Y && points[i].X > bottomRight.X))
            {
                bottomRight = points[i];
            }
        }

        // Identify remaining points based on their position relative to top-left and bottom-right
        var remaining = points.Where(p => p != topLeft && p != bottomRight).ToList();
        var topRight = points.Where(p => p.X > topLeft.X).OrderBy(p => p.Y).FirstOrDefault();
        var bottomLeft = remaining.FirstOrDefault(p => p != topRight);

        if (topRight == default || bottomLeft == default)
        {
            throw new ArgumentException("Input points do not form a valid rectangle.");
        }

        // Return points in clockwise order
        return [ topLeft, topRight, bottomRight, bottomLeft ];
    }
}

internal static class OBBNonMaxSuppressionHelper
{
    public static void TestIou()
    {
        var v1 = new OBBIndexedBoundingBox()
        {
            X = 0,
            Y = 0,
            Width = 4,
            Height = 2,
            Angle = 30,
            Index = 0,
            Class = new YoloV8Class(0, "test"),
            Confidence = 0.9f
        };

        var v2 = new OBBIndexedBoundingBox()
        {
            X = 1,
            Y = 1,
            Width = 3,
            Height = 3,
            Angle = 45,
            Index = 0,
            Class = new YoloV8Class(0, "test"),
            Confidence = 0.9f
        };
        var iou = v1.CalculateIoU(v2); // sb 0.3101366419488868
    }

    public static OBBIndexedBoundingBox[] Suppress(OBBIndexedBoundingBox[] boxes, float iouThreshold)
    {
        OBBNonMaxSuppressionHelper.TestIou();

        Array.Sort(boxes);

        var boxCount = boxes.Length;

        var activeCount = boxCount;

        var isNotActiveBoxes = new bool[boxCount];

        var selected = new List<OBBIndexedBoundingBox>();

        for (int i = 0; i < boxCount; i++)
        {
            if (isNotActiveBoxes[i])
            {
                continue;
            }

            var boxA = boxes[i];

            selected.Add(boxA);

            for (var j = i + 1; j < boxCount; j++)
            {
                if (isNotActiveBoxes[j])
                {
                    continue;
                }

                var boxB = boxes[j];

                if (CalculateIoU(boxA, boxB) > iouThreshold)
                {
                    isNotActiveBoxes[j] = true;

                    activeCount--;

                    if (activeCount <= 0)
                    {
                        break;
                    }
                }
            }

            if (activeCount <= 0)
            {
                break;
            }
        }

        return [.. selected];
    }

    private static double CalculateIoU(this OBBIndexedBoundingBox boxA, OBBIndexedBoundingBox boxB)
    {
        var areaA = Area(boxA);

        if (areaA <= 0f)
        {
            return 0f;
        }

        var areaB = Area(boxB);

        if (areaB <= 0f)
        {
            return 0f;
        }

        var vertices1 = boxA.GetCornerPoints();
        var vertices2 = boxB.GetCornerPoints();

        var rect1 = vertices1.Select(v => new IntPoint(v.X, v.Y)).ToList();
        var rect2 = vertices2.Select(v => new IntPoint(v.X, v.Y)).ToList();

        Clipper clipper = new Clipper();
        clipper.AddPath(rect1, PolyType.ptSubject, true);
        clipper.AddPath(rect2, PolyType.ptClip, true);

        List<List<IntPoint>> intersection = new List<List<IntPoint>>();
        clipper.Execute(ClipType.ctIntersection, intersection, PolyFillType.pftEvenOdd, PolyFillType.pftEvenOdd);

        List<List<IntPoint>> union = new List<List<IntPoint>>();
        clipper.Execute(ClipType.ctUnion, union, PolyFillType.pftEvenOdd, PolyFillType.pftEvenOdd);

        if (intersection.Count == 0 || union.Count == 0)
        {
            return 0f;
        }
        double intersectionArea = Clipper.Area(intersection[0]);
        double unionArea = Clipper.Area(union[0]);

        var iou = intersectionArea / unionArea;

        return iou;
    }

    private static int Area(OBBIndexedBoundingBox obb)
    {
        return (int)(obb.Width * obb.Height);
    }
}