from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/start_stream")
async def start_stream():
    video_path = '../data/videos/office_camera_test.mp4'
    # start_rtsp_stream(video_path)
    return {"message": "Started"}
