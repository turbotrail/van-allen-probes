#!/bin/bash

lftp https://spdf.gsfc.nasa.gov <<EOF
mirror \
    --parallel=8 \
    --continue \
    --only-newer \
    /pub/data/rbsp/rbspa/l2/emfisis/magnetometer/uvw/2015 \
    ./uvw/2015
EOF

echo "2015 download completed."