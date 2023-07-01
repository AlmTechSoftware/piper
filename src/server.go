package main

import (
	"io"
	"log"
	"net"
	"os"
	"sync"
)

func handleClient(conn net.Conn, wg *sync.WaitGroup, workers []**os.Process) {
	log.Println("New client connected:", conn.RemoteAddr().String())

	// Create piperworker
	worker := newPiperWorker()

	defer func() {
		conn.Close()
		wg.Done()
		log.Println("Client disconnected:", conn.RemoteAddr().String())
	}()

	// Create a buffer to hold the incoming frame data
	buffer := make([]byte, 1024)

	for {
		// Read frame data from the client
		n, err := conn.Read(buffer)
		if err != nil {
			if err == io.EOF {
				continue
			}
			log.Println("Error reading from client:", err)
			continue
		}

		// Send to PiperWorker for processing

		// Recieve data and return to client

		// Return new proc data to client
		// _, err = conn.Write(newFrameData)
		// if err != nil {
		// 	log.Println("Error sending frame-data back to client:", err)
		// 	return
		// }
	}
}
