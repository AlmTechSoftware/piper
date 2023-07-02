#!/bin/sh

# sources="main.go server.go"
sources=$(ls $PWD | grep -E ".go$")

go run $sources
