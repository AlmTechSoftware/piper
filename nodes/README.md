# PiperNodes

This is where all of the services goes.

    services
    ├── canvas
    ├── CDNN
    ├── CEMNN
    ├── ColorRemap
    ├── CRAD
    ├── SGNN
    └── vect

## Each "PiperNode"

Each "pipernode" should have a `node.yml` in its root containing the following format (example):

```yml
node:
  name: SGNN
  exec: "python"
```
