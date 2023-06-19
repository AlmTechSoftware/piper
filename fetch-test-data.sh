#!/usr/bin/bash

ytdl=yt-dlp

out=test-data

fetch() {
	mkdir -p $out
	while read src; do
		$ytdl $(echo $src | awk '{print $1}') -o $out/$(echo $src | awk '{print $2}')
	done <sources
}

fetch
