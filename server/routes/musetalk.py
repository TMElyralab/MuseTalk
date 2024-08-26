from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
from server.config import config

from server.dto import MuseTalkRequest, AvatarRequest

from scripts.realtime_inference import Avatar

router = APIRouter()
avatar = None

@router.post("/generate_musetalk_result/")
async def generate_musetalk_result(req: MuseTalkRequest) -> FileResponse:
    #TODO get avatar from db
    # avatar = await req["avatar_id"]
    print("Inferring using:", req.audio_path)
    output_vidname = req.avatar_id
    avatar.inference(
        req.audio_path, 
        output_vidname, 
        req.fps,
        req.skip_save_images
    )
    
    avatar_path = f"{config.MUSETALK_PATH}/results/avatars/{req.avatar_id}"
    video_out_path = f"{avatar_path}/vid_output/"
    vid_path = os.path.join(video_out_path, output_vidname+".mp4")
        
    response = FileResponse(vid_path)

    return response

@router.post("/create_musetalk_avatar/")
async def create_musetalk_avatar(req: AvatarRequest):
    global avatar
    avatar_id = str(uuid.uuid4())
    video_path = req.video_path
    bbox_shift = req.bbox_shift
    batch_size = req.batch_size
    avatar = Avatar(
        avatar_id = avatar_id, 
        video_path = video_path, 
        bbox_shift = bbox_shift, 
        batch_size = batch_size,
        preparation= True)
    #TODO save avatar to db
    #TODO add id to local storage or cookie
    return {avatar_id: avatar_id}

@router.post("/update_musetalk_avatar/")
async def update_musetalk_avatar(req)->None:
    ...
    
@router.post("delete_musetalk_avatar")
async def delete_musetalk_avatar(req)->None:
    ...

@router.get("get_musetalk_avatar")
async def get_musetalk_avatar(req)->None:
    ...