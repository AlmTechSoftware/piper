# Use the Ubuntu base image
FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
	cmake \
	coreutils \
	make \
	git \
	libswscale-dev \
	libavcodec-dev \
	libavutil-dev \
	ffmpeg \
	libsm6 \
	libxext6 \
    && rm -rf /var/lib/apt/lists/*



# Set the working directory
WORKDIR /opt/piper

# Copy the source code into the container
COPY src/ .
COPY requirements.txt .

# Copy the libraries into the container
COPY libs libs/

# Install the libraries
RUN pip3 install --break-system-packages -r requirements.txt

# Install the custom libraries
RUN for directory in libs/*; do \
        if [ -d "$directory" ]; then \
            pip3 install $directory; \
        fi; \
    done


# Create a shared volume for logs
VOLUME /var/log/piper

# Expose ws port
EXPOSE 4242

RUN mkdir -p /var/log/piper

CMD ["sh", "-c", "LOG_FILE=\"/var/log/piper/$(date +'%Y-%m-%d').log\" && echo \"Log file path: $LOG_FILE\" && touch \"$LOG_FILE\" && chmod 666 \"$LOG_FILE\" && python3 -u -m piper | tee -a \"$LOG_FILE\""]
