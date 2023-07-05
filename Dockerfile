# Use the Ubuntu base image
FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    systemd \
	libswscale-dev \
	libavcodec-dev \
	libavutil-dev \
    && rm -rf /var/lib/apt/lists/*



# Set the working directory
WORKDIR /opt/piper

# Copy the source code into the container
COPY src/ .
COPY requirements.txt .
COPY piper.service .

# Copy the libraries into the container
COPY libs libs/

# Install the libraries
RUN pip3 install -r requirements.txt

# Install the custom libraries
RUN for directory in libs/*; do \
        if [ -d "$directory" ]; then \
            pip3 install $directory; \
        fi; \
    done

# Copy the systemd service unit file
COPY piper.service /etc/systemd/system/piper.service

# Create a shared volume for logs
VOLUME /var/log/piper

# Enable the systemd service
RUN systemctl enable --now piper.service

# Start the systemd service
CMD ["/sbin/init"]
