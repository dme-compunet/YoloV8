using ClipperLib;
using Compunet.YoloV8.Data;
using Compunet.YoloV8.Metadata;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using System.Text;
using static Compunet.YoloV8.YoloV8Extensions;
//using static Compunet.YoloV8.YoloV8Extensions;

// https://github.com/ultralytics/ultralytics/issues/7667

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

            public required double X { get; init; }
            public required double Y { get; init; }
            public required double Width { get; init; }
            public required double Height { get; init; }
            public required double Angle { get; init; }

            public required float Confidence { get; init; }

            public int CompareTo(OBBIndexedBoundingBox other) => Confidence.CompareTo(other.Confidence);

            public SixLabors.ImageSharp.Point[] GetCornerPoints()
            {
                var angle = Angle * Math.PI / 180.0; // Radians
                var b = (float)Math.Cos(angle) * 0.5f;
                var a = (float)Math.Sin(angle) * 0.5f;

                var pt = new SixLabors.ImageSharp.Point[4];
                pt[0].X = (int)Math.Round(X - a * Height - b * Width, 0);
                pt[0].Y = (int)Math.Round(Y + b * Height - a * Width, 0);
                pt[1].X = (int)Math.Round(X + a * Height - b * Width, 0);
                pt[1].Y = (int)Math.Round(Y - b * Height - a * Width, 0);
                pt[2].X = (int)Math.Round(2 * X - pt[0].X, 0);
                pt[2].Y = (int)Math.Round(2 * Y - pt[0].Y, 0);
                pt[3].X = (int)Math.Round(2 * X - pt[1].X, 0);
                pt[3].Y = (int)Math.Round(2 * Y - pt[1].Y, 0);

                // Calculate the distances of each point from the origin (0, 0)
                double distance1 = Math.Sqrt(Math.Pow(pt[0].X, 2) + Math.Pow(pt[0].Y, 2));
                double distance2 = Math.Sqrt(Math.Pow(pt[1].X, 2) + Math.Pow(pt[1].Y, 2));
                // rotate if necessary to ensure pt[0] is the top-left point
                if (distance2 < distance1)
                {
                    var temp = pt[0];
                    pt[0] = pt[1];
                    pt[1] = pt[2];
                    pt[2] = pt[3];
                    pt[3] = temp;
                }
                return pt;
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

                    var x = (output[0, 0, i] - xPadding) * xRatio;
                    var y = (output[0, 1, i] - yPadding) * yRatio;

                    var w = output[0, 2, i] * xRatio;
                    var h = output[0, 3, i] * yRatio;
                    var a = output[0, detectionDataSize - 1, i]; // radians
                    //angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
                    if (a >= Math.PI && a <= 0.75 * Math.PI)
                    {
                        a -= (float)Math.PI;
                    }

                    var name = _metadata.Names[maxConfidenceIndex];

                    boxes[i] = new OBBIndexedBoundingBox
                    {
                        Index = i,
                        Class = name,
                        X = x,
                        Y = y,
                        Width = w,
                        Height = h,
                        Angle = a * 180 / Math.PI, // degrees
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

                var nms = OBBNonMaxSuppressionHelper.Suppress(topBoxes, configuration.IoU);
                return nms;
            }
        }
    }
}

namespace Compunet.YoloV8.Data
{

    public class OBBBoundingBox
    {
        public required YoloV8Class Class { get; init; }

        public required double X { get; init; }
        public required double Y { get; init; }
        public required double Width { get; init; }
        public required double Height { get; init; }
        public required double Angle { get; init; }

        public required float Confidence { get; init; }

        public override string ToString()
        {
            return $"{Class.Name} ({Confidence:N})";
        }

        public SixLabors.ImageSharp.Point[] GetCornerPoints()
        {
            var angle = Angle * Math.PI / 180.0; // Radians
            var b = (float)Math.Cos(angle) * 0.5f;
            var a = (float)Math.Sin(angle) * 0.5f;

            var pt = new SixLabors.ImageSharp.Point[4];
            pt[0].X = (int)Math.Round(X - a * Height - b * Width, 0);
            pt[0].Y = (int)Math.Round(Y + b * Height - a * Width, 0);
            pt[1].X = (int)Math.Round(X + a * Height - b * Width, 0);
            pt[1].Y = (int)Math.Round(Y - b * Height - a * Width, 0);
            pt[2].X = (int)Math.Round(2 * X - pt[0].X, 0);
            pt[2].Y = (int)Math.Round(2 * Y - pt[0].Y, 0);
            pt[3].X = (int)Math.Round(2 * X - pt[1].X, 0);
            pt[3].Y = (int)Math.Round(2 * Y - pt[1].Y, 0);

            // Calculate the distances of each point from the origin (0, 0)
            double distance1 = Math.Sqrt(Math.Pow(pt[0].X, 2) + Math.Pow(pt[0].Y, 2));
            double distance2 = Math.Sqrt(Math.Pow(pt[1].X, 2) + Math.Pow(pt[1].Y, 2));
            // rotate if necessary to ensure pt[0] is the top-left point
            if (distance2 < distance1)
            {
                var temp = pt[0];
                pt[0] = pt[1];
                pt[1] = pt[2];
                pt[2] = pt[3];
                pt[3] = temp;
            }
            return pt;
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

internal static class OBBNonMaxSuppressionHelper
{
    public static OBBIndexedBoundingBox[] Suppress(OBBIndexedBoundingBox[] boxes, float iouThreshold)
    {
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