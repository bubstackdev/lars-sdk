using System.Text.Json.Nodes;
using Moedex.Lars;

namespace Moedex.Lars.Tests;

public enum ImageColor
{
    [Option("Red")] Red,
    [Option("Blue")] Blue,
    [Option("Another color")] Other,
}

public sealed class ImageTests
{
    [Fact]
    public void InvalidImageSizesAreRejected()
    {
        Assert.Throws<LarsException>(() => ImageInput.FromBytes([]));
        Assert.Throws<LarsException>(() => ImageInput.FromBytes(new byte[20 * 1024 * 1024 + 1]));
    }

    [Fact]
    public async Task ImagesDoNotSilentlyRunOnTextPacks()
    {
        using var mock = new MockPack();
        var image = ImageInput.FromBytes([1, 2, 3]);
        var predicate = mock.Lars.Noul("visible", "Something is visible");
        var error = await Assert.ThrowsAsync<LarsException>(() => predicate.AskImageAsync(image));
        Assert.Equal("invalid_request", error.Type);
    }

    [Fact]
    public async Task RealImagesProduceTypedDecisions()
    {
        var pack = Environment.GetEnvironmentVariable("LARS_TEST_VISION_PACK");
        if (pack is null) return;
        using var lars = await Lars.OpenAsync(pack);
        // Fixtures are provided explicitly so this test also runs from packaged test outputs.
        var root = Environment.GetEnvironmentVariable("LARS_TEST_IMAGE_DIR")!;
        var red = await ImageInput.FromFileAsync(Path.Combine(root, "red.png"));
        var blue = await ImageInput.FromFileAsync(Path.Combine(root, "blue.png"));
        var color = lars.Choice<ImageColor>("color", "What is the predominant color in the image?");
        Assert.Equal(ImageColor.Red, (await color.AskImageAsync(red)).Value);
        var decision = await color.AskImageAsync(blue);
        Assert.Equal(ImageColor.Blue, decision.Value);
        Assert.False(decision.Calibrated);
        var visible = lars.Noul("red", "The image is predominantly red");
        var answers = await lars.AskImageAsync(red, new Predicate<string>[] { visible, color });
        Assert.NotNull(answers);
    }
}
