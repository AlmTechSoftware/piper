package main

import (
	"log"

	"github.com/gorilla/websocket"
)

func handlePiper(conn *websocket.Conn, workers []*PiperWorker) {
	log.Println("New client connected:", conn.RemoteAddr().String())

	// Create piperworker
	worker := newPiperWorker()
	defer worker.kill()

	// Append worker to workers
	workers = append(workers, worker)

	for {
		_, frameData, err := conn.ReadMessage()
		if err != nil {
			log.Println("Failed to read WS message:", err)
			continue
		}

		// TODO: async

		// Send the frame data to the worker
		worker.send(frameData)

		// Now recv data from the worker and send back to client
		frameData, err = worker.recv(BUFFER_SIZE)
		if err != nil {
			log.Println("Frame data processing failed:", err)
		}

		// Returned the processed frames to the client
		err = conn.WriteMessage(1, frameData)
		if err != nil {
			log.Println("Unable to send data back to client:", err)
		}
	}
}
