# Freeze-Omni

Project address: [Freeze-Omni Repository](https://github.com/VITA-MLLM/Freeze-Omni) 

---

## Deployment Guide

### 1. Successfully deploy the Freeze-Omni
Ensure that the Freeze-Omni project has been successfully deployed and all dependencies are installed.

### 2. Place the server file
After deployment, copy `server_streaming.py` into the following path:

`/your/path/Freeze-Omni/bin`

### 3. Modify the demo server script
Open and edit:

`/your/path/Freeze-Omni/scripts/run_demo_server.sh`

Find this section:

```bash
CUDA_VISIBLE_DEVICES=1 python3 bin/server.py \
```

Change server.py → server_streaming.py:

```bash
CUDA_VISIBLE_DEVICES=1 python3 bin/server_streaming.py \
```

### 4. Place the demo_stream.html
Copy the demo_stream.html file into:

`/your/path/Freeze-Omni/web/resources`

### 5. Start the backend server
Run the demo script to launch the backend

### 6. Open the frontend
Open the frontend web page in your browser.
The address is usually: 

`https://<your-server-ip>:<port>`

### 7. Run real-time inference

Open another terminal and run:

```
bash realtime_inference.sh
```

This script will feed audio input into the system and generate real-time outputs.
