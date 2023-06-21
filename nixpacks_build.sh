#!/bin/sh

name=piper-worker
path=.


nixpacks build $path \
			--name $name \
			--start-cmd "python -m piper"
