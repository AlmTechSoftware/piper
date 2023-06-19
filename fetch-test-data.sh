#!/usr/bin/bash

ytdl=yt-dlp

out=tests

fetch() {
	mkdir -p $out
	while read src; do
		$ytdl $(echo $src | awk '{print $1}') -o $out/$(echo $src | awk '{print $2}')
	done <sources.list
}

fetch
