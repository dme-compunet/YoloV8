namespace Compunet.YoloV8.Extensions;

public static class ImageSharpContoursExtensions
{
    private static readonly (Func<Point, Point> func, int neighborhood)[] _neighborhood;

    static ImageSharpContoursExtensions()
    {
        _neighborhood = new (Func<Point, Point>, int)[]
        {
            (point => new Point(point.X - 1, point.Y), 7),
            (point => new Point(point.X - 1, point.Y - 1), 7),
            (point => new Point(point.X, point.Y - 1), 1),
            (point => new Point(point.X + 1, point.Y - 1), 1),
            (point => new Point(point.X + 1, point.Y), 3),
            (point => new Point(point.X + 1, point.Y + 1), 3),
            (point => new Point(point.X, point.Y+1), 5),
            (point => new Point(point.X -1, point.Y + 1), 5)
        };
    }

    public static IReadOnlyList<IReadOnlyList<Point>> GetContours(this Image image)
    {
        var luminance = image.CloneAs<L8>();

        var found = new HashSet<Point>();

        bool inside = false;

        var contours = new List<IReadOnlyList<Point>>();

        for (int y = 0; y < luminance.Height; y++)
            for (int x = 0; x < luminance.Width; x++)
            {
                Point point = new(x, y);

                if (found.Contains(point) && !inside)
                {
                    inside = true;
                    continue;
                }

                bool transparent = IsTransparent(luminance, point);

                if (!transparent && inside)
                    continue;

                if (transparent && inside)
                {
                    inside = false;
                    continue;
                }

                if (!transparent && !inside)
                {
                    var contour = new List<Point>();

                    contours.Add(contour);

                    found.Add(point);
                    contour.Add(point);

                    int checkLocationNr = 1;
                    Point startPos = point;

                    int counter1 = 0;
                    int counter2 = 0;

                    while (true)
                    {
                        Point checkPosition = _neighborhood[checkLocationNr - 1].func(point);

                        int newCheckLocationNr = _neighborhood[checkLocationNr - 1].neighborhood;

                        if (!IsTransparent(luminance, checkPosition))
                        {
                            if (checkPosition == startPos)
                            {
                                counter1++;

                                if (newCheckLocationNr == 1 || counter1 >= 3)
                                {
                                    inside = true;
                                    break;
                                }
                            }

                            checkLocationNr = newCheckLocationNr;
                            point = checkPosition;
                            counter2 = 0;
                            found.Add(point);
                            contour.Add(point);
                        }
                        else
                        {
                            checkLocationNr = 1 + (checkLocationNr % 8);

                            if (counter2 > 8)
                                break;
                            else
                                counter2++;
                        }
                    }
                }
            }

        return contours;
    }

    private static bool IsTransparent(Image<L8> image, Point pixel)
    {
        return pixel.X > image.Width - 1
               || pixel.X < 0
               || pixel.Y > image.Height - 1
               || pixel.Y < 0
               || image[pixel.X, pixel.Y].PackedValue == 0;
    }
}