#!/usr/bin/env bash
# leg.sh — the ONE way to run a model load on this APU box (BC-250, 15 GiB unified).
# Enforces vulkan/PORT-NOTES laws 10 + 11: the crashes of 2026-10-26/27/28 were every case of
# a model load escaping these rules. Do not run engines or reference dumps directly; run them
# through this.
#
#   tools/vulkan/leg.sh --gate engine  <command...>    # omnistrata-gguf-vk: own arena, ~14.1 GB,
#                                                      #   safe if it is the ONLY big load
#   tools/vulkan/leg.sh --gate cpu     <command...>    # llama.cpp --ngl 0 + STRATA_DUMP_MMAP=1:
#                                                      #   weights as reclaimable page cache
#   tools/vulkan/leg.sh --gate gpu     <command...>    # llama.cpp -ngl>0: pins 12+ GB through TTM,
#                                                      #   needs 12.3 GiB free — effectively
#                                                      #   headless-only; refuses with a clear
#                                                      #   reason on a desktop box
#
# Rules applied: a MemAvailable floor before launch (in MB — the 1024-vs-1048576 units bug cost
# a whole day), ONE big load at a time (a lockfile with the live pid), choom -n 900 so the leg
# dies before any session process, nice -n 10, and a 20 s cooldown note (caller sleeps between
# legs; the lock serializes the legs themselves).
set -u
LOCK=/tmp/us2-leg.lock
gate=""
case "${1:-}" in
  --gate) gate="$2"; shift 2;;
  *) echo "usage: leg.sh --gate engine|cpu|gpu <command...>"; exit 2;;
esac
case "$gate" in
  engine) floor=10500;;   # the engine maps its own arena; measured safe at 13.9-14.1 GB resident
  cpu)    floor=9400;;    # mmap page cache is reclaimable: 9.2 GiB available is plenty
  gpu)    floor=12300;;   # pinned TTM tiers: ONLY a machine idle enough for 12.0 GiB free
  *) echo "leg.sh: unknown gate '$gate'"; exit 2;;
esac
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
# one big load at a time: take the lock, reaping a stale holder
if ! ( set -o noclobber; echo "$$ $*" > "$LOCK" ) 2>/dev/null; then
  holder=$(awk '{print $1}' "$LOCK" 2>/dev/null || echo "")
  if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
    echo "leg.sh: another leg holds the lock (pid $holder): $(cut -c1-120 <(sed -n 1p "$LOCK" | cut -d' ' -f2-))" >&2
    exit 3
  fi
fi
trap 'rm -f "$LOCK"' EXIT
while [ "$(avail)" -lt "$floor" ]; do
  echo "leg.sh: gate '$gate' needs ${floor} MB available, have $(avail) — waiting" >&2
  sleep 30
done
env -u STRATA_DUMP_MMAP bash -c "STRATA_DUMP_MMAP=1 choom -n 900 -- nice -n 10 $*"
rc=$?
echo "leg.sh: rc=$rc avail_after=$(avail)MB $(date +%H:%M:%S)"
exit $rc
