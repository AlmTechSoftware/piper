#!/bin/sh

name=piper-worker
path=./src


nixpacks build $path \
			--name $name \
			--build-cmd $path/build.sh \
			--start-cmd "python -m piperworker"
