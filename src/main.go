package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"strconv"
	"sync"
	// "google.golang.org/grpc"
)

func startWorkers(numWorkers int, frameChan chan []byte) []*exec.Cmd {
	workers := make([]*exec.Cmd, numWorkers)
	for i := 0; i < numWorkers; i++ {
		cmd := exec.Command("python", "worker.py") // Change the command and arguments accordingly if needed
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		// Create a named pipe (FIFO) for each worker
		fifoName := fmt.Sprintf("fifo%d", i)
		err := exec.Command("mkfifo", fifoName).Run()
		if err != nil {
			log.Println("Error creating named pipe:", err)
			return nil
		}

		// Connect the named pipe to the worker process' standard input
		fifo, err := os.OpenFile(fifoName, os.O_WRONLY, os.ModeNamedPipe)
		if err != nil {
			log.Println("Error opening named pipe:", err)
			return nil
		}
		cmd.ExtraFiles = append(cmd.ExtraFiles, fifo)

		workers[i] = cmd

		go func() {
			err := cmd.Run()
			if err != nil {
				log.Println("Worker process exited with error:", err)
			}
		}()
	}

	return workers
}

func main() {
	// Parse .env stuff
	var port = os.Getenv("PORT")
	if port == "" {
		port = "4242"
	}

	var workerCountStr = os.Getenv("WORKER_COUNT")
	var workerCount, err = strconv.Atoi(workerCountStr)
	if err != nil {
		log.Fatalln("Unable to read worker count from environment!")
	}

	// Start a TCP server to accept client connections
	lis, err := net.Listen("tcp", ":"+port)

	if err != nil {
		log.Println("Error starting server:", err)
		return
	}

	defer lis.Close()

	log.Printf("Piper Server running on port %s", port)
	log.Println("Waiting for connections...")

	var wg sync.WaitGroup

	for {
		// accept a new client connection
		conn, err := lis.Accept()
		if err != nil {
			log.Println("Error accepting client connection:", err)
			continue
		}

		// Handle the client connection asynchronously
		wg.Add(1)
		go handleClient(conn, &wg)
	}

	// NOTE: Will never reach but good to have I guess?
	wg.Wait()
	log.Println("Server shutting down...")

	// // Create gRPC server
	// grpcServer := grpc.NewServer()
	//
	// // Register your gRPC service
	// videoStreamSvc := &videoStreamServer{}
	// RegisterVideoStreamServiceServer(grpcServer, videoStreamSvc)

	// // Start gRPC server
	// log.Println("Starting gRPC server...")
	// if err := grpcServer.Serve(lis); err != nil {
	// 	log.Fatalf("Failed to serve: %v", err)
	// }
}
