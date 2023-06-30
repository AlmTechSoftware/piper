package main

import (
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"sync"
	"syscall"
)

func createWorker(id int) (*os.Process, string, error) {
	socket := fmt.Sprintf("/tmp/piperworker_%d.socket", id)

	// Remove the socket path if exist
	os.Remove(socket)

	// Create the socket file
	os.Create(socket)  // FIXME: perm error

	// Start the PiperWorker
	worker := exec.Command("python", "-m", "piperworker", socket)
	worker.Stdout = os.Stdout
	worker.Stderr = os.Stderr

	err := worker.Start()
	if err != nil {
		log.Fatalln("Failed to start worker with error:", err)
		return nil, socket, err
	}

	return worker.Process, socket, err
}

func handleClient(conn net.Conn, wg *sync.WaitGroup, frameChan channel) {
	log.Println("New client connected:", conn.RemoteAddr().String())
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
		frameData := buffer[:n]
		frameChan <- frameData
		log.Printf("Received frame from %s: %v\n", conn.RemoteAddr().String(), frameData)

		// Recieve data and return to client
		newFrameData := <-frameChan

		// Return new proc data to client
		_, err = conn.Write(newFrameData)
		if err != nil {
			log.Println("Error sending frame-data back to client:", err)
			return
		}
	}
}
