// Molar Triage in C#: sort a dental practice's inbox with three predicates, then calibrate them on
// 30 labeled messages and measure on 30 others. The messages are fictional (examples/molar_triage).
//
//     LARS_NATIVE=<path to liblars> dotnet run -- <pack>

using System.Text.Json.Nodes;

using Moedex.Lars;

var data = Path.Combine(AppContext.BaseDirectory, "../../../../../../examples/molar_triage");
using var lars = await Lars.OpenAsync(args.Length > 0 ? args[0] : "lars-4b");

var inPain = lars.Noul("inPain", "The sender says they are in pain right now.");
var department = lars.Choice<Department>("department", "Which department should take this message?");
var urgency = lars.Score("urgency", "How urgently does this message need attention?",
    "Routine: can wait for the next regular visit",
    "Soon: book within a few weeks",
    "This week: schedule in the next few days",
    "Today: needs same-day attention",
    "Emergency: go to the practice or an emergency room now");

// The predicates are functions, so the triage reads like the policy it implements.
async Task<string> Triage(string message)
{
    var r = await lars.AskAsync(message, inPain, department, urgency);
    if (r.Of(urgency).Level >= 4) return "call the patient now";
    if (r.Of(inPain) && r.Of(urgency).Level >= 3) return $"same-day slot with {r.Of(department).Value}";
    return $"queue for {r.Of(department).Value}";
}

Console.WriteLine(await Triage("I knocked out my front tooth playing hockey twenty minutes ago and it's bleeding a lot."));
Console.WriteLine(await Triage("Can I get a copy of my last invoice for my insurance?"));

// Each labeled row is one message and one question; group them per predicate.
List<(string State, string Instructions, string Label)> Rows(string file) =>
    File.ReadLines(Path.Combine(data, file)).Select(line => JsonNode.Parse(line))
        .Select(r => (r!["state"]!.GetValue<string>(), r["question"]!["instructions"]!.GetValue<string>(), r["label"]!.GetValue<string>()))
        .ToList();
var calibration = Rows("molar_triage.calibration.jsonl");
var test = Rows("molar_triage.test.jsonl");
const string Pain = "The sender says they are in pain right now.";
const string Dept = "Which department should take this message?";
const string Urg = "How urgently does this message need attention?";
static Department Parse(string key) => Enum.Parse<Department>(key.Replace("_", ""), ignoreCase: true);

async Task<string> Accuracy()
{
    var pain = await Task.WhenAll(test.Where(r => r.Instructions == Pain).Select(async r => (await inPain.AskAsync(r.State)).Value == (r.Label == "yes")));
    var dept = await Task.WhenAll(test.Where(r => r.Instructions == Dept).Select(async r => (Department)await department.AskAsync(r.State) == Parse(r.Label)));
    var urg = await Task.WhenAll(test.Where(r => r.Instructions == Urg).Select(async r => (await urgency.AskAsync(r.State)).Level == int.Parse(r.Label)));
    string Share(bool[] hits) => $"{hits.Count(h => h)}/{hits.Length}";
    return $"in pain {Share(pain)}, department {Share(dept)}, urgency {Share(urg)}";
}

Console.WriteLine($"\ntest accuracy, pack calibrator:       {await Accuracy()}");
await inPain.CalibrateAsync(calibration.Where(r => r.Instructions == Pain).Select(r => (r.State, r.Label == "yes")));
await department.CalibrateAsync(calibration.Where(r => r.Instructions == Dept).Select(r => (r.State, Parse(r.Label))));
await urgency.CalibrateAsync(calibration.Where(r => r.Instructions == Urg).Select(r => (r.State, int.Parse(r.Label))));
Console.WriteLine($"test accuracy, calibrated on 30 rows: {await Accuracy()}");

enum Department
{
    [Option("Cleanings, checkups, whitening, sensitivity advice")] Hygiene,
    [Option("Fillings, crowns, chipped or broken teeth, lost fillings")] Restorative,
    [Option("Braces, aligners, retainers, wires")] Orthodontics,
    [Option("Extractions, wisdom teeth, implants, post-operative care")] Surgery,
    [Option("Insurance, invoices, payment plans, estimates")] Billing,
    [Option("Scheduling, forms, hours, directions, records requests")] FrontDesk,
}
