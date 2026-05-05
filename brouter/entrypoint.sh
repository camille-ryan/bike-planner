#!/bin/sh
# Symlink custom profiles into BRouter's profiles2 directory so its URL
# lookup (?profile=name → profiles2/name.brf) finds them. The third CLI
# arg to RouteServer ("customprofiles") only handles uploaded profiles,
# not files dropped on disk by us.
set -e

for f in /opt/brouter/customprofiles/*.brf; do
    [ -f "$f" ] || continue
    ln -sfn "$f" "/opt/brouter/profiles2/$(basename "$f")"
done

exec "$@"
