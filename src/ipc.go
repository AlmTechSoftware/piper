package main

import (
	"io"
	"log"
	"net"
)

func reader(r io.Reader, size int) ([]byte, error) {
	buf := make([]byte, size)
	n, err := r.Read(buf[:])
	if err != nil {
		return nil, err
	}

	return buf[0:n], err
}

func sender(socket string, data []byte) (int, error) {
	c, err := net.Dial("unix", socket)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	n, err := c.Write(data)
	if err != nil {
		log.Println("IPC socket write error:", err)
	}

	return n, err
}
