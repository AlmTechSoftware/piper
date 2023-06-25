package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"bufio"
	// "google.golang.org/grpc"
)

type channel = chan []byte

func startWorkerProcs(numWorkers int, frameChan channel, resultChan channel) []*exec.Cmd {
	workers := make([]*exec.Cmd, numWorkers)
	for i := 0; i < numWorkers; i++ {
		cmd := exec.Command("python", "worker.py") // Change the command and arguments accordingly if needed
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		// Connect the worker process' standard input to the frame data channel
		stdin, err := cmd.StdinPipe()
		if err != nil {
			log.Println("Error creating stdin pipe for worker process:", err)
			return nil
		}
		go func() {
			for frameData := range frameChan {
				_, err := stdin.Write(frameData)
				if err != nil {
					log.Println("Error writing to worker process:", err)
					return
				}
			}
		}()

		// Connect the worker process' standard output to the result channel
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			log.Println("Error creating stdout pipe for worker process:", err)
			return nil
		}
		go func() {
			reader := bufio.NewReader(stdout)
			for {
				resultData, err := reader.ReadBytes('\n')
				if err != nil {
					log.Println("Error reading result data from worker process:", err)
					return
				}
				resultChan <- resultData
			}
		}()

		// Start the worker process
		err = cmd.Start()
		if err != nil {
			log.Println("Error starting worker process:", err)
			return nil
		}

		workers[i] = cmd
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

	// Start the workers
	var wg sync.WaitGroup
	frameChan := make(chan []byte)
	workers := startWorkerProcs(workerCount, frameChan)

	if workers == nil {
		log.Fatalln("Failed to start workers.")
	}

	defer func() {
		for _, wp := range workers {
			wp.Process.Kill()
		}
	}()

	log.Printf("Piper Server running on port %s", port)
	log.Println("Waiting for connections...")

	for {
		// accept a new client connection
		conn, err := lis.Accept()
		if err != nil {
			log.Println("Error accepting client connection:", err)
			continue
		}

		// Handle the client connection asynchronously
		wg.Add(1)
		go handleClient(conn, &wg, frameChan)
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
