# Piper

Microservices & algorithm for CHALK

## Basic overview

Here is the basic overview of how the pipeline algorithm works:

- Canvas Detection Network (**CDN**)
- Canvas rotation & angle demorpher (**CRAD**) 
- Convex Hull & Canvas Cleanup (basic contrast & convex hull algo.)
- Contour Extraction Model (neural net) (**CEM**)
- Skeleton Generator (**SG**)
- Vectorizer
- (Color Remap)

## Platform

Should be able to run on a modren GNU/Linux system inside of a container (docker version >= 24.0.2). 

## Processing Format

The algorithm should follow a "microservice" structure where load-balancing is implemented. Each worker should be able to process multiple
frames at a time for different people but a load-balancer should redirect work to other new workers if overloaded.

![MS overview](/img/overview.png)

## The Goal

The algorithm should work and run on a reasonable CPU and each "frame" should take at most 150ms to process. The algorithm should follow a "best-effort" approach where **_minor_** artifacts are considered okay

## Contributing

Allways make a [**pull-request**](https://github.com/AlmTechSoftware/piper/pulls) and **NEVER PUSH TO THE MAIN/PRODUCTION BRANCH**. Each pull-request should have an issue associated with it.

 - Code should not be commented, just remove it.
 - Do not leave debug code.
 - Do not abbreviate variable, function, and method -names.

### Formatting

Follow language standards like [PEP8](https://peps.python.org/pep-0008/) for Python and [Style Guidelines](https://doc.rust-lang.org/1.0.0/style/README.html) for Rust etc.

 - TS/JS, JSON, YML, ...: [Prettier](https://prettier.io/)
 - Rust: [rustfmt](https://github.com/rust-lang/rustfmt)
 - Python: [black](https://github.com/psf/black)

Otherwise use **one tab** and set the **tab width** yourself in order to avoid indentation width debates.

## LICENSE

See [**LICENSE**](/LICENSE). 

Copyright 2023 (c) Elias Almqvist
