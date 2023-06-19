# Chalk Prototype 1

## Usage
`./app.py (input video path: path)`

## Algorithm

### (-1) Frame Detection
Detect the frame of the board and extract its vertices.

### (A) Canvas Extraction
Given the vertices of the board, do a anti-perspective transform on
the image to extract the canvas of the board.

### (AA) Remove Humans
Remove humans.

### (B) Contour 
Given the canvas, calculate the background color and extract the contours of
the contents of the canvas.

### (C) Skeleton 
Given the contours: find the medial axis for each in a short given interval
proportional to the inputted resolution.

### (D) Bezier
Given the skelton lines. Perform and create bezier curves and render
the contents on a new frame.

### (EEE) Cleanup
Remove any unwanted results using a classifier (clutter removal)

## License
Copyright 2023 - Elias Almqvist
