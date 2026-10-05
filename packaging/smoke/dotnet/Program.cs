using System.Runtime.InteropServices;

using Moedex.Lars;

Environment.SetEnvironmentVariable("LARS_NATIVE", null);
var dir = Directory.CreateTempSubdirectory("lars-nuget-smoke-").FullName;
File.WriteAllText(Path.Combine(dir, "pack.json"), """{"format": 1, "name": "smoke", "version": "0.0.0", "backend": "mock", "template": "plain"}""");
using var lars = await Lars.OpenAsync(dir);
var isRefund = lars.Noul("isRefund", "The customer asks for a refund");
var team = lars.Choice<Team>("team", "Which team?");
bool refund = await isRefund.AskAsync("refund me");
Team picked = await team.AskAsync("I was charged twice");
Console.WriteLine($"ok: Moedex.Lars {Lars.Version} on {RuntimeInformation.RuntimeIdentifier}: {refund} {picked}");

enum Team { [Option("Charges")] Billing, [Option("Bugs")] Technical }
