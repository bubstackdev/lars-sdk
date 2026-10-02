#!/bin/sh
# The whole Phase 2b sweep, unattended: both queues in parallel, kept awake by caffeinate, a
# report after every run and a final one at the end. Started by launchd
# (~/Library/LaunchAgents/dev.lars-sdk.sweep.plist), so it outlives terminals and sessions and
# resumes after a reboot; it unloads itself when everything is done.
#
#     launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.lars-sdk.sweep.plist   # start
#     launchctl bootout gui/$(id -u)/dev.lars-sdk.sweep                                 # stop
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
echo "sweep started $(date)"
sweep/queue.sh metal >> sweep/out/queue-metal.log 2>&1 &
sweep/queue.sh cpu >> sweep/out/queue-cpu.log 2>&1 &
wait
sweep/report.sh
echo "sweep finished $(date)"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.sweep" 2>/dev/null
