// Moedex.Lars on a mock pack (no model: deterministic, keyword-driven answers that test the
// plumbing, not judgment), and on a real pack when LARS_TEST_PACK names one. The native library is
// found through LARS_NATIVE (CI sets it to the cargo build).

using System.Text.Json.Nodes;

using Moedex.Lars;

namespace Moedex.Lars.Tests;

public enum Team
{
    [Option("Payments, charges, refunds")] Billing,
    [Option("Bugs, outages, errors")] Technical,
    [Option("Scheduling, hours, records", Key = "desk")] FrontDesk,
}

public enum Tag
{
    [Option("Wants money back")] Refund,
    [Option("Reports a defect")] Bug,
}

public sealed record Ticket(string Subject, string Body, string Email);

public sealed class MockPack : IDisposable
{
    public MockPack()
    {
        Dir = Directory.CreateTempSubdirectory("lars-dotnet-").FullName;
        File.WriteAllText(Path.Combine(Dir, "pack.json"),
            """{"format": 1, "name": "dotnet", "version": "0.0.0", "backend": "mock", "template": "plain"}""");
        Lars = Lars.Open(new LarsOptions { Pack = Dir, Threads = 2 });
    }

    public string Dir { get; }
    public Lars Lars { get; }

    public void Dispose() => Lars.Dispose();
}

public sealed class MockPackTests(MockPack mock) : IClassFixture<MockPack>
{
    private Lars Lars => mock.Lars;

    [Fact]
    public void OpensAndReports()
    {
        Assert.Equal("dotnet", Lars.Pack["name"]!.GetValue<string>());
        Assert.Matches(@"^\d+\.\d+\.\d+", Lars.Version);
        Assert.True(Lars.Status()["loaded"]!.GetValue<bool>());
    }

    [Fact]
    public async Task NoulsAreBooleansEverywhere()
    {
        var isRefund = Lars.Noul("isRefund", "The customer asks for a refund");
        bool blocking = isRefund.Check("Please refund my order");
        NoulDecision d = await isRefund.AskAsync("Please refund my order");
        Assert.Equal(blocking, (bool)d);
        Assert.Equal(d.P >= 0.5, d.Value);
        Assert.Equal(Math.Max(d.P, 1 - d.P), d.Confidence, 10);
        if (await isRefund.AskAsync("Please refund my order")) Assert.True(d.Value);
        Func<string, bool> asFunc = isRefund;
        Assert.Equal(blocking, asFunc("Please refund my order"));
    }

    [Fact]
    public async Task ChoicesAreEnums()
    {
        var team = Lars.Choice<Team>("team", "Which team should handle this?");
        Team picked = team.Pick("I was charged twice for my payment");
        Team awaited = await team.AskAsync("I was charged twice for my payment");
        Assert.Equal(picked, awaited);
        var d = await team.AskAsync("I was charged twice for my payment");
        Assert.Equal([Team.Billing, Team.Technical, Team.FrontDesk], d.Probabilities.Keys.Order());
        Assert.Equal(d.Probabilities.Values.Max(), d.Confidence);
        var criteria = team.Question["criteria"]!.AsObject().Select(kv => kv.Key);
        Assert.Equal(["billing", "technical", "desk"], criteria); // snake_case names, or the [Option] key
    }

    [Fact]
    public async Task ScoresAndMultis()
    {
        var urgency = Lars.Score("urgency", "How urgent?", "routine", "soon", "today", "emergency");
        var level = urgency.Rate("The server is down");
        Assert.InRange(level, 0, 3);
        var d = await urgency.AskAsync("The server is down");
        Assert.Equal(4, d.Probabilities.Count);
        var tags = Lars.Multi<Tag>("tags", "Which apply?");
        var selected = tags.Select("refund my money, the app has a bug");
        Assert.All(selected, t => Assert.True(t is Tag.Refund or Tag.Bug));
    }

    [Fact]
    public async Task RecordsNeedAStateSelectorThatPicksFields()
    {
        JsonNode? seen = null;
        var urgent = Lars.Noul("ticketUrgent", "The ticket is urgent", new NoulOptions<Ticket>
        {
            State = t => seen = new JsonObject { ["subject"] = t.Subject, ["body"] = t.Body },
        });
        await urgent.AskAsync(new Ticket("Down", "Everything is down", "someone@example.com"));
        Assert.Equal("""{"subject":"Down","body":"Everything is down"}""", seen!.ToJsonString());
        var noSelector = Lars.Noul("ticketNoSelector", "The ticket is urgent", new NoulOptions<Ticket>());
        var error = await Assert.ThrowsAsync<LarsException>(() => noSelector.AskAsync(new Ticket("a", "b", "c")));
        Assert.Equal("invalid_request", error.Type);
    }

    [Fact]
    public async Task AskPutsSeveralPredicatesToOneRequest()
    {
        var angry = Lars.Noul("angry", "The customer is angry");
        var team = Lars.Choice<Team>("askTeam", "Which team?");
        var answers = await Lars.AskAsync("I was charged twice and I am furious", angry, team);
        _ = answers.Of(angry).Value;
        Assert.True(Enum.IsDefined(answers.Of(team).Value));
    }

    [Fact]
    public async Task ManyConcurrentCallsAllComplete()
    {
        var isRefund = Lars.Noul("isRefund", "The customer asks for a refund");
        var results = await Task.WhenAll(Enumerable.Range(0, 300).Select(i => isRefund.AskAsync($"message {i}: refund please")));
        Assert.Equal(300, results.Length);
        var blocking = await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
            Enumerable.Range(0, 25).Select(i => isRefund.Check($"thread call {i}")).Count())));
        Assert.All(blocking, n => Assert.Equal(25, n));
    }

    [Fact]
    public async Task CancellationCancelsQueuedCalls()
    {
        var isRefund = Lars.Noul("isRefund", "The customer asks for a refund");
        using var cancel = new CancellationTokenSource();
        var calls = Enumerable.Range(0, 400).Select(i => isRefund.AskAsync($"call {i}", cancel.Token)).ToList();
        cancel.Cancel();
        var outcomes = await Task.WhenAll(calls.Select(async c =>
        {
            try { await c; return "ok"; }
            catch (OperationCanceledException) { return "cancelled"; }
        }));
        Assert.Equal(400, outcomes.Length);
        Assert.All(outcomes, o => Assert.True(o is "ok" or "cancelled"));
    }

    [Fact]
    public async Task CalibrationIsPerPredicateAndSavesAndLoads()
    {
        var isRefund = Lars.Noul("calibrated", "The customer asks for a refund");
        var examples = Enumerable.Range(0, 40).Select(i => (i % 2 == 1 ? $"refund order {i}" : $"hello {i}", i % 2 == 1));
        var fit = await isRefund.CalibrateAsync(examples, maxError: 0.05);
        Assert.Equal(40, fit.Examples);
        Assert.Null(fit.Certified!.Threshold);
        Assert.Contains("cannot certify", fit.Certified.Reason);
        // Certification was asked for and failed: the gate stays shut.
        Assert.False((await isRefund.AskAsync("Please refund my order")).Accepted);
        Assert.False(isRefund.Check("Please refund my order"));
        var saved = Lars.Calibrations();
        Assert.NotNull(saved["calibrators"]!["calibrated"]);

        using var other = Lars.Open(mock.Dir);
        other.LoadCalibrations(saved);
        Assert.Contains("calibrated", other.Status()["calibrators"]!.AsArray().Select(n => n!.GetValue<string>()));
    }

    [Fact]
    public async Task AnUncertifiedChoiceFallsBack()
    {
        var team = Lars.Choice<string, Team>("fallbackTeam", "Which team?", new ChoiceOptions<string, Team> { Fallback = Team.FrontDesk });
        var inputs = Enumerable.Range(0, 100).Select(i => $"message {i} about something").ToList();
        var labels = new List<Team>();
        foreach (var x in inputs) labels.Add((await team.AskAsync(x)).Value);
        var fit = await team.CalibrateAsync(inputs.Zip(labels), maxError: 0.0001);
        Assert.Null(fit.Certified!.Threshold);
        Assert.False((await team.AskAsync("anything")).Accepted);
        Assert.Equal(Team.FrontDesk, team.Pick("anything"));
    }

    [Fact]
    public async Task SavedCalibrationsCarryTheirQuestionAndCertificate()
    {
        var refund = Lars.Noul("savedRefund", "The customer asks for a refund");
        await refund.CalibrateAsync(Enumerable.Range(0, 40).Select(i => ($"refund order {i}", i % 3 != 0)), maxError: 0.05);
        var saved = Lars.Calibrations();
        Assert.Equal(2, saved["format"]!.GetValue<int>());
        Assert.Equal(refund.Question.ToJsonString(), saved["questions"]!["savedRefund"]!.ToJsonString());

        // Loaded elsewhere, the calibration is bound to its question.
        using var other = Lars.Open(mock.Dir);
        other.LoadCalibrations(saved);
        Assert.Throws<LarsException>(() => other.Noul("savedRefund", "The customer is NOT asking for a refund"));
        var same = other.Noul("savedRefund", "The customer asks for a refund");
        Assert.False((await same.AskAsync("refund me")).Accepted);

        // Defined first, the other question makes the load fail, and nothing changes.
        using (var third = Lars.Open(mock.Dir))
        {
            third.Noul("savedRefund", "The customer is NOT asking for a refund");
            Assert.Throws<LarsException>(() => third.LoadCalibrations(saved));
            Assert.Empty(third.Status()["calibrators"]!.AsArray());
        }

        // A replacement without a certificate clears the old one.
        var certified = saved.DeepClone().AsObject();
        certified["certified"] = new JsonObject { ["savedRefund"] = new JsonObject { ["maxError"] = 0.05, ["threshold"] = 0.5 } };
        other.LoadCalibrations(certified);
        Assert.Equal(0.5, other.Calibrations()["certified"]!["savedRefund"]!["threshold"]!.GetValue<double>());
        var uncertified = saved.DeepClone().AsObject();
        uncertified["certified"] = new JsonObject();
        other.LoadCalibrations(uncertified);
        Assert.Null(other.Calibrations()["certified"]!["savedRefund"]);
        Assert.True((await same.AskAsync("refund me")).Accepted);

        // A bad entry anywhere in the file changes nothing.
        var bad = certified.DeepClone().AsObject();
        bad["calibrators"]!["broken"] = new JsonObject { ["kind"] = "nonsense" };
        bad["questions"]!["broken"] = new JsonObject { ["type"] = "noul", ["instructions"] = "x" };
        Assert.Throws<LarsException>(() => other.LoadCalibrations(bad));
        Assert.Null(other.Calibrations()["certified"]!["savedRefund"]);
        Assert.Null(other.Calibrations()["calibrators"]!["broken"]);
        var badCertificate = certified.DeepClone().AsObject();
        badCertificate["certified"]!["savedRefund"]!["threshold"] = "high";
        Assert.Throws<LarsException>(() => other.LoadCalibrations(badCertificate));
        Assert.Null(other.Calibrations()["certified"]!["savedRefund"]);

        // Older files and other packs are refused.
        var old = saved.DeepClone().AsObject();
        old["format"] = 1;
        Assert.Throws<LarsException>(() => other.LoadCalibrations(old));
        var elsewhere = saved.DeepClone().AsObject();
        elsewhere["pack"]!["name"] = "another";
        Assert.Throws<LarsException>(() => other.LoadCalibrations(elsewhere));
    }

    [Fact]
    public async Task CalibrationsAreBoundToTheStateVersion()
    {
        var urgent = Lars.Noul<Ticket>("versionedUrgent", "The ticket needs a reply today",
            new NoulOptions<Ticket> { State = t => new JsonObject { ["subject"] = t.Subject }, StateVersion = "subject-only" });
        await urgent.CalibrateAsync(Enumerable.Range(0, 20).Select(_ => (new Ticket("Down", "All down", "a@example.com"), true)));
        var saved = Lars.Calibrations();
        Assert.Equal("subject-only", saved["stateVersions"]!["versionedUrgent"]!.GetValue<string>());

        using (var same = Lars.Open(mock.Dir))
        {
            same.LoadCalibrations(saved);
            Assert.Throws<LarsException>(() => same.Noul<Ticket>("versionedUrgent", "The ticket needs a reply today",
                new NoulOptions<Ticket> { State = t => new JsonObject { ["body"] = t.Body }, StateVersion = "body-only" }));
            Assert.Throws<LarsException>(() => same.Noul("versionedUrgent", "The ticket needs a reply today"));
            same.Noul<Ticket>("versionedUrgent", "The ticket needs a reply today",
                new NoulOptions<Ticket> { State = t => new JsonObject { ["subject"] = t.Subject }, StateVersion = "subject-only" });
        }

        using var other = Lars.Open(mock.Dir);
        other.Noul<Ticket>("versionedUrgent", "The ticket needs a reply today",
            new NoulOptions<Ticket> { State = t => new JsonObject { ["body"] = t.Body }, StateVersion = "body-only" });
        Assert.Throws<LarsException>(() => other.LoadCalibrations(saved));
    }

    [Fact]
    public async Task PullReturnsAPackDirectory()
    {
        Assert.Equal(mock.Dir, await Lars.PullAsync(mock.Dir));
        Assert.Equal("pack_error", Assert.Throws<LarsException>(() => Lars.Pull("./no/such/pack")).Type);
    }

    [Fact]
    public void ErrorsAreTyped()
    {
        Assert.Equal("pack_error", Assert.Throws<LarsException>(() => Lars.Open("./no/such/pack")).Type);
        Lars.Noul("unique", "One question");
        Lars.Noul("unique", "One question"); // the same question again is fine
        Assert.Throws<LarsException>(() => Lars.Noul("unique", "Another question"));
        var error = Assert.Throws<LarsException>(() => Lars.Evaluate(new JsonObject { ["state"] = "x", ["questions"] = new JsonObject() }));
        Assert.Equal("invalid_request", error.Type);
    }

    [Fact]
    public async Task CallsAfterDisposeFail()
    {
        var closing = Lars.Open(mock.Dir);
        var p = closing.Noul("p", "Anything");
        closing.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => p.AskAsync("x"));
    }
}

public sealed class RealPackTests
{
    private static readonly string? Pack = Environment.GetEnvironmentVariable("LARS_TEST_PACK");

    [Fact]
    public async Task JudgesPlainCasesTheWayAPersonWould()
    {
        if (Pack is null) return; // set LARS_TEST_PACK to a real pack to run
        using var lars = await Lars.OpenAsync(Pack);
        var isPissed = lars.Noul("isPissed", "The customer is angry");
        Assert.True(isPissed.Check("This is the third time your app ate my order. I am done with you people."));
        Assert.False(await isPissed.AskAsync("Thanks so much, the new update works great!"));
        var team = lars.Choice<Team>("team", "Which team should handle this message?");
        Assert.Equal(Team.Billing, team.Pick("I was charged twice on my credit card this month"));
        Assert.Equal(Team.Technical, (Team)await team.AskAsync("The app crashes every time I open settings"));
    }
}
