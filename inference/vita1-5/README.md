# VITA 1.5

Project address: [VITA 1.5 Repository](https://github.com/VITA-MLLM/VITA) 

---

## Deployment Guide

### 1. Successfully deploy the VITA 1.5
Ensure that the VITA 1.5 project has been successfully deployed and all dependencies are installed.

### 2. Place the server file
After deployment, copy `server_streaming.py` into the following path:

`/your/path/VITA-main/web_demo`

### 3. Place the demo_stream.html
Copy the demo_stream.html file into:

`/your/path/VITA-main/web_demo/vita_html/web/resources`

### 4. Start the backend server
Run the demo script to launch the backend

### 5. Open the frontend
Open the frontend web page in your browser.
The address is usually: 

`https://<your-server-ip>:<port>`

### 6. Run real-time inference

Open another terminal and run:

```
bash realtime_inference.sh
```


This script will feed audio input into the system and generate real-time outputs.
