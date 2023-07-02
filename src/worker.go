package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"syscall"

	"github.com/google/uuid"
	"github.com/james-barrow/golang-ipc"
)

type PiperWorker struct {
	id     uuid.UUID
	socket string
	ipc    *ipc.Client
	proc   *exec.Cmd
}

func newPiperWorker() *PiperWorker {
	// Generate uuid (name) of the worker
	var id = uuid.New()

	// Create the socket path
	var socket = fmt.Sprintf("/tmp/piperworker_%s.socket", id.String())

	// Remove the socket path if exist
	os.Remove(socket)

	// NOTE: PiperWorker (python) will create the IPC socket
	// // Create the socket file
	// os.Create(socket)

	// Start the PiperWorker
	proc := exec.Command("python", "-m", "piperworker", socket)
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr

	err := proc.Start()
	if err != nil {
		log.Fatalln("Failed to start worker with error:", err)
		return nil
	}

	// Create a client IPC connection
	c_ipc, err := ipc.StartClient(socket, nil)
	if err != nil {
		log.Println("Error occurred when create client IPC for piperworker.", err)
		return nil
	}

	// Create the actual worker object
	piperWorker := PiperWorker{id, socket, c_ipc, proc}

	return &piperWorker
}

func (w PiperWorker) send(data []byte) error {
	err := w.ipc.Write(1, data) // TODO: msgType?
	return err
}

func (w PiperWorker) recv(size int) ([]byte, error) {
	msg, err := w.ipc.Read()
	return msg.Data, err
}

func (w PiperWorker) kill() {
	w.proc.Process.Signal(syscall.SIGTERM)
	w.proc.Process.Wait()
	w.ipc.Close()
}
