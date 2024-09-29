#nullable disable

namespace YoloV8.Tests;

public class UseCudaDefaultValueTest
{
    [Fact]
    public void UseCuda_Default_Value_Should_Match_Configuration()
    {
        var gpuValue = BuildAndGetDefaultValue(true);
        var noGpuValue = BuildAndGetDefaultValue(false);

        Assert.True(gpuValue);
        Assert.False(noGpuValue);
    }

    private static bool BuildAndGetDefaultValue(bool gpu)
    {
        const string Folder = "test";
        const string Project = "../../../../YoloV8/YoloV8.csproj";

        var configuration = gpu ? "GpuRelease" : "Release";
        var library = gpu ? "YoloV8.Gpu.dll" : "YoloV8.dll";

        var process = Process.Start("dotnet", $"build {Project} -c {configuration} -o {Folder}/");

        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException("Build failed");
        }

        var assembly = Assembly.LoadFile(Path.GetFullPath($"./{Folder}/{library}"));
        var optionsInfo = assembly.GetType(typeof(YoloPredictorOptions).FullName);
        var useCudaInfo = optionsInfo.GetProperty(nameof(YoloPredictorOptions.UseCuda));

        var optionsInstance = Activator.CreateInstance(optionsInfo);

        return (bool)useCudaInfo.GetValue(optionsInstance);
    }
}
