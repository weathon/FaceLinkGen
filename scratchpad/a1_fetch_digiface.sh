#!/bin/bash
# DigiFace-1M 500K part -> /raid/wg25r/digiface
# Resumable: curl -C - on partial zips, unzip -n never overwrites, .done sentinel per zip.
set -e
BASE=https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023
D=/raid/wg25r/digiface
mkdir -p $D/zips $D/images

for Z in subjects_100000-133332_5_imgs subjects_133333-166665_5_imgs subjects_166666-199998_5_imgs; do
    if [ -f $D/zips/$Z.done ]; then echo "=== $Z already extracted, skip"; continue; fi
    echo "=== download $Z $(date)"
    curl -f -L -C - -o $D/zips/$Z.zip $BASE/$Z.zip
    echo "=== test $Z $(date)"
    unzip -t -q $D/zips/$Z.zip
    echo "=== extract $Z $(date)"
    unzip -n -q $D/zips/$Z.zip -d $D/images
    touch $D/zips/$Z.done
    rm $D/zips/$Z.zip
done
echo "=== ALL DONE $(date)"
