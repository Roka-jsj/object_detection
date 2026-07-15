ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-core AS deps

# Install system dependencies early for better caching
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-pip \
    python3-rosdep \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Create ros2_ws and copy files
WORKDIR /root/ros2_ws
COPY . /root/ros2_ws/src

# Install ROS dependencies
RUN rosdep init && rosdep update --include-eol-distros
RUN apt update && rosdep install --from-paths src --ignore-src -r -y && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Ubuntu 22.04 (humble/iron) ships pip 22.0.2, which has no PEP 658 support and so
# downloads every candidate wheel while backtracking to satisfy numpy<2. It does not
# converge. Upgrade pip on that branch before resolving requirements.
RUN if [ "$(lsb_release -rs | cut -d. -f1)" -ge 24 ]; then \
    pip3 install -r src/requirements.txt --break-system-packages --ignore-installed --no-cache-dir; \
    else \
    pip3 install --no-cache-dir --upgrade pip && \
    python3 -m pip install -r src/requirements.txt --ignore-installed --no-cache-dir; \
    fi

FROM deps AS builder

SHELL ["/bin/bash", "-c"]

# Build the workspace
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

# Source the ROS 2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Run a default command, e.g., starting a bash shell
CMD ["bash"]
