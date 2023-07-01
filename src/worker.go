package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/google/uuid"
)

type PiperWorker struct {
	id uuid.UUID
	socket string
	proc *exec.Cmd
}


func newPiperWorker() *PiperWorker {
	// Generate uuid (name) of the worker
	var id = uuid.New()

	// Create the socket path
	var socket = fmt.Sprintf("/tmp/piperworker_%d.socket", id.String())

	// Remove the socket path if exist
	os.Remove(socket)

	// Create the socket file
	os.Create(socket) // FIXME: perm error

	// Start the PiperWorker
	proc := exec.Command("python", "-m", "piperworker", socket)
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr

	err := proc.Start()
	if err != nil {
		log.Fatalln("Failed to start worker with error:", err)
		return nil
	}

	piperWorker := PiperWorker{id, socket, proc}

	return &piperWorker
}

