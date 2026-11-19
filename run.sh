#!/bin/bash
log=$1
shift
for i in 1 2 3 4 5 6 7 8; do
    "$@" >> "$log" 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
        sleep 60
        [ $(($(date +%s) - $(stat -c %Y "$log"))) -gt 1500 ] && echo "stalled, restarting" >> "$log" && kill -9 $pid
    done
    wait $pid && exit 0
done
exit 1
