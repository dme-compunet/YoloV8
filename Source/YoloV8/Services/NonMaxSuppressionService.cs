namespace Compunet.YoloV8.Services;

internal class NonMaxSuppressionService : INonMaxSuppressionService
{
    public T[] Suppress<T>(Span<T> boxes, float iouThreshold) where T : IRawBoundingBox<T>
    {
        if (boxes.Length == 0)
        {
            return [];
        }

        // Sort by confidence from the high to the low 
        boxes.Sort((x, y) => y.CompareTo(x));

        // Initialize result with highest confidence box
        var result = new List<T>(4)
        {
            boxes[0]
        };

        // Iterate boxes (Skip with the first box because it already has been added)
        for (var i = 1; i < boxes.Length; i++)
        {
            var box1 = boxes[i];
            var addToResult = true;

            for (var j = 0; j < result.Count; j++)
            {
                var box2 = result[j];

                // Skip boxers with different label
                if (box1.Name != box2.Name)
                {
                    continue;
                }

                // If the box overlaps another box already in the results 
                if (T.CalculateIoU(ref box1, ref box2) > iouThreshold)
                {
                    addToResult = false;
                    break;
                }
            }

            if (addToResult)
            {
                result.Add(box1);
            }
        }

        return [.. result];
    }
}